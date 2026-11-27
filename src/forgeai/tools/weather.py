"""Weather forecast from open-meteo (keyless). The model never supplies a URL: this code
builds every request itself, to two fixed hosts, from validated arguments."""

from __future__ import annotations

from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, Field

from forgeai.schemas.requirements import ToolHint
from forgeai.tools.base import ToolEnv, ToolResult, expect_args

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
ALLOWED_HOSTS = frozenset(
    {"geocoding-api.open-meteo.com", "api.open-meteo.com", "marine-api.open-meteo.com"}
)
DAILY = (
    "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max"
)


class WeatherArgs(BaseModel):
    location: str = Field(min_length=2, max_length=80, description="Place name, e.g. 'Kochi'.")
    days: int = Field(default=3, ge=1, le=7, description="Forecast days (1-7).")
    marine: bool = Field(default=False, description="Also return wave height (coastal places).")


async def _get_json(client: httpx.AsyncClient, url: str, params: dict[str, Any]) -> Any:
    if httpx.URL(url).host not in ALLOWED_HOSTS:  # defence in depth
        raise ValueError(f"host not allowed: {url}")
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


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
        try:
            geo = await _get_json(
                client, GEOCODE_URL, {"name": args.location, "count": 1, "format": "json"}
            )
            places = geo.get("results") or []
            if not places:
                return ToolResult.fail(f"place not found: {args.location!r}")
            place = places[0]
            lat, lon = float(place["latitude"]), float(place["longitude"])
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
            waves: list[Any] = []
            if args.marine:
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
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            return ToolResult.fail(f"weather service unavailable ({type(exc).__name__})")
        finally:
            if owns_client:
                await client.aclose()

        daily = forecast.get("daily", {})
        lines = [
            f"Forecast for {place.get('name')}, {place.get('country', '')} ({lat:.2f}, {lon:.2f}):"
        ]
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
        return ToolResult(ok=True, output="\n".join(lines), data={"daily": daily, "waves": waves})
