"""Weather forecast from open-meteo (keyless), with MET Norway + OpenStreetMap Nominatim as
a fallback when open-meteo refuses (its free tier rate-limits by IP, and shared cloud
hosts hit that limit). The model never supplies a URL: this code builds every request
itself, to fixed hosts, from validated arguments."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, Field

from forgeai.schemas.requirements import ToolHint
from forgeai.tools.base import ToolEnv, ToolResult, expect_args

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
METNO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
ALLOWED_HOSTS = frozenset(
    {
        "geocoding-api.open-meteo.com",
        "api.open-meteo.com",
        "marine-api.open-meteo.com",
        "nominatim.openstreetmap.org",
        "api.met.no",
    }
)
# MET Norway and Nominatim require an identifying User-Agent.
HEADERS = {"User-Agent": "ForgeAI/0.1 (+https://github.com/Amitreddy14/forgeai)"}
RETRY_STATUS = {429, 500, 502, 503, 504}
MS_TO_KN = 1.943844
RETRY_DELAY_S = 0.5
DAILY = (
    "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max"
)


class WeatherArgs(BaseModel):
    location: str = Field(
        min_length=2,
        max_length=80,
        description="Place name WITH its country, e.g. 'Kochi, India' (many names exist twice).",
    )
    days: int = Field(default=3, ge=1, le=7, description="Forecast days (1-7).")
    marine: bool = Field(default=False, description="Also return wave height (coastal places).")


class ServiceError(Exception):
    """A weather host failed; the message names the host and the status."""


async def _get_json(
    client: httpx.AsyncClient, url: str, params: dict[str, Any], *, retries: int = 2
) -> Any:
    host = httpx.URL(url).host
    if host not in ALLOWED_HOSTS:  # defence in depth
        raise ValueError(f"host not allowed: {url}")
    for attempt in range(retries + 1):
        try:
            response = await client.get(url, params=params, headers=HEADERS)
        except httpx.HTTPError as exc:
            if attempt == retries:
                raise ServiceError(f"{host}: {type(exc).__name__}") from exc
        else:
            if response.status_code == 200:
                return response.json()
            if response.status_code not in RETRY_STATUS or attempt == retries:
                raise ServiceError(f"{host}: HTTP {response.status_code}")
        await asyncio.sleep(RETRY_DELAY_S * (attempt + 1))
    raise ServiceError(f"{host}: no response")  # pragma: no cover


async def _geocode(client: httpx.AsyncClient, name: str, country: str) -> dict[str, Any] | None:
    """{name, country, lat, lon} or None. Open-meteo first, Nominatim if it refuses."""
    try:
        geo = await _get_json(client, GEOCODE_URL, {"name": name, "count": 10, "format": "json"})
    except ServiceError:
        query = f"{name}, {country}" if country else name
        found = await _get_json(
            client, NOMINATIM_URL, {"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1}
        )
        if not found:
            return None
        top = found[0]
        return {
            "name": name.title(),
            "country": (top.get("address") or {}).get("country", country.title()),
            "lat": float(top["lat"]),
            "lon": float(top["lon"]),
        }
    places = geo.get("results") or []
    if country:
        # "Kochi" alone resolves to Kochi, Japan first; the country disambiguates.
        places = [
            p
            for p in places
            if country
            in {str(p.get("country", "")).lower(), str(p.get("country_code", "")).lower()}
        ]
    if not places:
        return None
    top = places[0]
    return {
        "name": top.get("name"),
        "country": top.get("country", ""),
        "lat": float(top["latitude"]),
        "lon": float(top["longitude"]),
    }


def _open_meteo_lines(daily: dict[str, Any], waves: list[Any]) -> list[str]:
    lines = []
    for i, day in enumerate(daily.get("time", [])):
        parts = [
            f"{day}: {daily['temperature_2m_min'][i]}-{daily['temperature_2m_max'][i]} °C",
            f"rain {daily['precipitation_sum'][i]} mm",
            f"wind {daily['wind_speed_10m_max'][i]} kn",
            f"gusts {daily['wind_gusts_10m_max'][i]} kn",
        ]
        if i < len(waves) and waves[i] is not None:
            parts.append(f"waves {waves[i]} m")
        lines.append(" | ".join(parts))
    return lines


def _metno_lines(data: dict[str, Any], days: int) -> list[str]:
    """Fold MET Norway's hourly series into daily lines (UTC dates)."""
    temps: dict[str, list[float]] = defaultdict(list)
    winds: dict[str, list[float]] = defaultdict(list)
    rain: dict[str, float] = defaultdict(float)
    for step in (data.get("properties") or {}).get("timeseries", []):
        day = str(step.get("time", ""))[:10]
        details = ((step.get("data") or {}).get("instant") or {}).get("details") or {}
        if "air_temperature" in details:
            temps[day].append(float(details["air_temperature"]))
        if "wind_speed" in details:
            winds[day].append(float(details["wind_speed"]))
        hour = (step.get("data") or {}).get("next_1_hours") or {}
        rain[day] += float((hour.get("details") or {}).get("precipitation_amount", 0.0))
    lines = []
    for day in sorted(temps)[:days]:
        lines.append(
            f"{day}: {min(temps[day]):.1f}-{max(temps[day]):.1f} °C | rain {rain[day]:.1f} mm"
            f" | wind {max(winds[day] or [0.0]) * MS_TO_KN:.1f} kn"
        )
    return lines


class Weather:
    name: ClassVar[str] = "weather"
    capability: ClassVar[ToolHint] = "weather"
    description: ClassVar[str] = (
        "Daily weather forecast for a named place (temperature, rain, wind, gusts; "
        "optionally wave height) from open-meteo."
    )
    args_model: ClassVar[type[BaseModel]] = WeatherArgs

    async def run(self, args: BaseModel, env: ToolEnv) -> ToolResult:
        args = expect_args(args, WeatherArgs)
        owns_client = env.http is None
        client = env.http or httpx.AsyncClient(timeout=10.0)
        notes: list[str] = []
        try:
            name, _, country = args.location.partition(",")
            name, country = name.strip(), country.strip().lower()
            place = await _geocode(client, name, country)
            if place is None:
                where = f" in {country.title()}" if country else ""
                return ToolResult.fail(f"place not found: {name!r}{where}")
            lat, lon = place["lat"], place["lon"]
            source = "open-meteo"
            daily: dict[str, Any] = {}
            waves: list[Any] = []
            try:
                forecast = await _get_json(
                    client,
                    FORECAST_URL,
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "daily": DAILY,
                        "timezone": "auto",
                        "forecast_days": args.days,
                        "wind_speed_unit": "kn",
                    },
                )
                daily = forecast.get("daily", {})
                lines = _open_meteo_lines(daily, [])
            except ServiceError as exc:
                notes.append(f"open-meteo refused ({exc}); used MET Norway")
                source = "MET Norway"
                metno = await _get_json(
                    client, METNO_URL, {"lat": round(lat, 4), "lon": round(lon, 4)}
                )
                lines = _metno_lines(metno, args.days)
            if args.marine:
                try:
                    marine = await _get_json(
                        client,
                        MARINE_URL,
                        {
                            "latitude": lat,
                            "longitude": lon,
                            "daily": "wave_height_max",
                            "timezone": "auto",
                            "forecast_days": args.days,
                        },
                    )
                    waves = marine.get("daily", {}).get("wave_height_max", [])
                    if daily:
                        lines = _open_meteo_lines(daily, waves)
                except ServiceError as exc:
                    notes.append(f"wave heights unavailable ({exc})")
        except ServiceError as exc:
            return ToolResult.fail(f"weather service unavailable ({exc})")
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            return ToolResult.fail(f"weather service unavailable ({type(exc).__name__})")
        finally:
            if owns_client:
                await client.aclose()

        if not lines:
            return ToolResult.fail("weather service returned no forecast")
        header = (
            f"Forecast for {place['name']}, {place['country']} ({lat:.2f}, {lon:.2f}),"
            f" source {source}:"
        )
        output = "\n".join([header, *lines, *(f"Note: {n}" for n in notes)])
        return ToolResult(
            ok=True, output=output, data={"daily": daily, "waves": waves, "source": source}
        )
