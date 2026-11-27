from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from forgeai.schemas.corpus import Corpus, Document, Page
from forgeai.schemas.requirements import Requirements, Topic
from forgeai.tools import REGISTRY, ToolEnv, ToolProposal, call_tool, select_tools, tool_card
from forgeai.tools.calculator import CalcError, safe_eval
from forgeai.tools.csv_lookup import matches

BERTHS = [
    {"berth_id": "B1", "port": "Kochi", "max_loa_m": "250", "daily_rate_usd": "4,200"},
    {"berth_id": "B2", "port": "Kochi", "max_loa_m": "180", "daily_rate_usd": "2600"},
    {"berth_id": "M4", "port": "Mumbai", "max_loa_m": "300", "daily_rate_usd": "6100"},
]


def env(**kwargs: Any) -> ToolEnv:
    return ToolEnv(tables={"berths": BERTHS}, **kwargs)


# ---------- Calculator ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "value"),
    [
        ("2 + 3 * 4", 14),
        ("(4200 * 3) * 1.18", 14868),
        ("1,500 * 2", 3000),
        ("round(10 / 3, 2)", 3.33),
        ("max(250, 180) - min(3, 7)", 247),
        ("2 ** 10", 1024),
        ("-5 + abs(-2)", -3),
        ("ceil(4.2) + floor(4.8)", 9),
    ],
)
def test_calculator_evaluates(expr: str, value: float) -> None:
    assert safe_eval(expr) == pytest.approx(value)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('dir')",
        "open('x')",
        "x + 1",
        "(1).__class__",
        "[1, 2]",
        "lambda: 1",
        "1 / 0",
        "10 ** 1000",
        "9" * 300,
        "",
        "True + 1",
        "round(1, n=2)",
    ],
)
def test_calculator_refuses_anything_but_arithmetic(expr: str) -> None:
    with pytest.raises(CalcError):
        safe_eval(expr)


async def test_calculator_tool_returns_errors_as_results() -> None:
    tool = REGISTRY["calculator"]
    ok = await call_tool(tool, {"expression": "3 * 4200"}, env())
    assert ok.ok
    assert ok.output == "3 * 4200 = 12600"
    bad = await call_tool(tool, {"expression": "os.system('x')"}, env())
    assert not bad.ok
    assert bad.output.startswith("ERROR")
    missing = await call_tool(tool, {}, env())
    assert "invalid arguments" in missing.output


# ---------- CSV lookup ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("cell", "cond", "expected"),
    [
        ("250", ">=250", True),
        ("4,200", ">4000", True),
        ("180", ">=250", False),
        ("Kochi", "kochi", True),
        ("Kochi Port", "~port", True),
        ("Kochi", "!=Mumbai", True),
        ("Kochi", ">A", False),
    ],
)
def test_filter_conditions(cell: str, cond: str, expected: bool) -> None:
    assert matches(cell, cond) is expected


async def test_csv_lookup_filters_and_projects() -> None:
    result = await call_tool(
        REGISTRY["csv_lookup"],
        {
            "table": "berths",
            "filters": {"PORT": "kochi", "max_loa_m": ">=200"},
            "columns": ["berth_id", "daily_rate_usd"],
        },
        env(),
    )
    assert result.ok
    assert result.data["rows"] == [{"row": 1, "berth_id": "B1", "daily_rate_usd": "4,200"}]


async def test_csv_lookup_explains_bad_table_and_column() -> None:
    tool = REGISTRY["csv_lookup"]
    bad_table = await call_tool(tool, {"table": "ships"}, env())
    assert "available: berths" in bad_table.output
    bad_col = await call_tool(tool, {"table": "berths", "filters": {"draft": "<10"}}, env())
    assert "columns are: berth_id" in bad_col.output


async def test_csv_lookup_limits_rows() -> None:
    result = await call_tool(REGISTRY["csv_lookup"], {"table": "berths", "limit": 1}, env())
    assert "3 matching row(s) in berths (showing first 1)" in result.output


# ---------- Current date --------------------------------------------------------------


async def test_current_date_uses_injected_clock() -> None:
    fixed = datetime(2026, 11, 28, 3, 30, tzinfo=UTC)
    result = await call_tool(REGISTRY["current_date"], {}, env(now=lambda: fixed))
    assert result.data == {"date": "2026-11-28", "weekday": "Saturday"}
    assert "09:00 (Asia/Kolkata)" in result.output
    bad = await call_tool(REGISTRY["current_date"], {"timezone": "Mars/Base"}, env())
    assert not bad.ok


# ---------- Weather (mocked HTTP) -----------------------------------------------------


def weather_transport(seen: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "geocoding-api.open-meteo.com":
            if request.url.params["name"] == "Atlantis":
                return httpx.Response(200, json={})
            # Like the real API: the Japanese Kochi comes first.
            japan = {
                "name": "Kochi",
                "country": "Japan",
                "country_code": "JP",
                "latitude": 33.56,
                "longitude": 133.53,
            }
            india = {
                "name": "Kochi",
                "country": "India",
                "country_code": "IN",
                "latitude": 9.93,
                "longitude": 76.26,
            }
            return httpx.Response(200, json={"results": [japan, india]})
        if request.url.host == "api.open-meteo.com":
            return httpx.Response(
                200,
                json={
                    "daily": {
                        "time": ["2026-11-28"],
                        "temperature_2m_min": [24.1],
                        "temperature_2m_max": [31.0],
                        "precipitation_sum": [2.5],
                        "wind_speed_10m_max": [12.0],
                        "wind_gusts_10m_max": [20.0],
                    }
                },
            )
        if request.url.host == "marine-api.open-meteo.com":
            return httpx.Response(200, json={"daily": {"wave_height_max": [1.4]}})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def test_weather_geocodes_then_forecasts_with_marine() -> None:
    seen: list[str] = []
    async with httpx.AsyncClient(transport=weather_transport(seen)) as client:
        result = await call_tool(
            REGISTRY["weather"],
            {"location": "Kochi, India", "days": 1, "marine": True},
            env(http=client),
        )
    assert result.ok, result.output
    assert "2026-11-28: 24.1-31.0 °C | rain 2.5 mm | wind 12.0 kn" in result.output
    assert "waves 1.4 m" in result.output
    assert "Forecast for Kochi, India (9.93, 76.26)" in result.output
    assert seen == [
        "geocoding-api.open-meteo.com",
        "api.open-meteo.com",
        "marine-api.open-meteo.com",
    ]


@pytest.fixture(autouse=True)
def _fast_weather_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("forgeai.tools.weather.RETRY_DELAY_S", 0.0)


async def test_weather_falls_back_to_met_norway_when_open_meteo_refuses() -> None:
    seen: list[str] = []
    metno = {
        "properties": {
            "timeseries": [
                {
                    "time": "2026-11-28T06:00:00Z",
                    "data": {
                        "instant": {"details": {"air_temperature": 25.0, "wind_speed": 5.0}},
                        "next_1_hours": {"details": {"precipitation_amount": 0.4}},
                    },
                },
                {
                    "time": "2026-11-28T09:00:00Z",
                    "data": {
                        "instant": {"details": {"air_temperature": 30.5, "wind_speed": 6.0}},
                        "next_1_hours": {"details": {"precipitation_amount": 1.1}},
                    },
                },
            ]
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        assert request.headers["user-agent"].startswith("ForgeAI/")
        if request.url.host.endswith("open-meteo.com"):
            return httpx.Response(429)
        if request.url.host == "nominatim.openstreetmap.org":
            assert request.url.params["q"] == "Kochi, india"
            return httpx.Response(
                200, json=[{"lat": "9.93", "lon": "76.26", "address": {"country": "India"}}]
            )
        return httpx.Response(200, json=metno)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await call_tool(
            REGISTRY["weather"],
            {"location": "Kochi, India", "days": 1, "marine": True},
            env(http=client),
        )
    assert result.ok, result.output
    assert "Kochi, India (9.93, 76.26), source MET Norway" in result.output
    assert "2026-11-28: 25.0-30.5 °C | rain 1.5 mm | wind 11.7 kn" in result.output
    assert "HTTP 429" in result.output  # the note says why the fallback was used
    assert "api.met.no" in seen


async def test_weather_outage_names_host_and_status() -> None:
    down = httpx.MockTransport(lambda _r: httpx.Response(403))
    async with httpx.AsyncClient(transport=down) as client:
        out = await call_tool(REGISTRY["weather"], {"location": "Kochi, India"}, env(http=client))
    assert not out.ok
    assert "HTTP 403" in out.output


async def test_weather_unknown_place_and_outage() -> None:
    async with httpx.AsyncClient(transport=weather_transport([])) as client:
        missing = await call_tool(REGISTRY["weather"], {"location": "Atlantis"}, env(http=client))
    assert "place not found" in missing.output
    down = httpx.MockTransport(lambda _r: httpx.Response(503))
    async with httpx.AsyncClient(transport=down) as client:
        out = await call_tool(REGISTRY["weather"], {"location": "Kochi"}, env(http=client))
    assert "unavailable" in out.output


def test_tool_cards_describe_arguments() -> None:
    card = tool_card(REGISTRY["csv_lookup"])
    assert card.startswith("- csv_lookup:")
    assert '"filters"' in json.dumps(card) or "filters" in card


# ---------- Tool-need contracts -------------------------------------------------------


def _req(brief: str, *hints: str) -> Requirements:
    topics = [Topic(topic="Leave rules")] + [
        Topic(topic=f"topic {h}", answered_from="tool", tool_hints=[h])  # type: ignore[arg-type]
        for h in hints
    ]
    return Requirements(
        agent_name="Test agent",
        purpose="p",
        scope="s",
        audience="a",
        tone="t",
        must_answer=topics,
        must_refuse=[{"topic": "Legal advice"}],  # type: ignore[list-item]
        brief=brief,
    )


def _corpus(with_csv: bool) -> Corpus:
    docs = [
        Document(
            doc_id="policy",
            filename="p.md",
            kind="md",
            title="P",
            pages=[Page(number=1, text="text")],
        )
    ]
    if with_csv:
        docs.append(
            Document(
                doc_id="berths",
                filename="b.csv",
                kind="csv",
                title="b",
                pages=[Page(number=1, text="Row 1: x: 1")],
            )
        )
    return Corpus(root=".", documents=docs)


def test_hr_agent_gets_no_weather_tool_even_if_proposed() -> None:
    req = _req("Answer HR policy questions for shore staff.")
    sel = select_tools(
        [
            ToolProposal(name="weather", justification="staff travel"),
            ToolProposal(name="current_date"),
            ToolProposal(name="teleport"),
        ],
        req,
        _corpus(False),
        [],
    )
    assert sel.attached == []
    reasons = {r.name: r.reason for r in sel.rejected}
    assert reasons["weather"] == "no requirement topic needs this capability"
    assert reasons["teleport"] == "not in the tool registry"


def test_hint_without_human_evidence_is_rejected() -> None:
    # The model put a weather topic into requirements, but the requester never asked.
    req = _req("Answer HR policy questions for shore staff.", "weather")
    sel = select_tools([ToolProposal(name="weather")], req, _corpus(False), [])
    assert sel.attached == []
    assert sel.rejected[0].reason == "the requester's own words never ask for it"


def test_port_agent_gets_exactly_what_it_needs() -> None:
    brief = (
        "Help port agents plan a call: find a berth that fits the vessel, estimate "
        "port dues, and check the wind forecast for the arrival window."
    )
    req = _req(brief, "table_lookup", "arithmetic", "weather")
    sel = select_tools(
        [ToolProposal(name="csv_lookup", justification="berths")], req, _corpus(True), []
    )
    assert [t.name for t in sel.attached] == ["calculator", "csv_lookup", "weather"]
    assert sorted(sel.added) == ["calculator", "weather"]
    evidence = {t.name: t.evidence for t in sel.attached}
    assert evidence["weather"] == 'requester wrote "forecast"'
    assert "berths" in evidence["csv_lookup"]


def test_requester_answers_count_as_evidence_but_csv_is_required() -> None:
    req = _req("Help port agents.", "current_date", "table_lookup")
    sel = select_tools([], req, _corpus(False), ["We need days until the berth booking expires"])
    assert [t.name for t in sel.attached] == ["current_date"]


# ---------- Phase 6.1: capabilities inferred from topic text ---------------------------


def _port_corpus() -> Corpus:
    berths = Document(
        doc_id="berths",
        filename="berths.csv",
        kind="csv",
        title="berths",
        columns=["berth_id", "port", "max_loa_m", "max_draft_m", "daily_berth_hire_usd"],
        pages=[Page(number=1, text="Row 1: berth_id: KOC-Q1")],
    )
    notices = Document(
        doc_id="notices",
        filename="n.pdf",
        kind="pdf",
        title="N",
        pages=[Page(number=1, text="text")],
    )
    return Corpus(root=".", documents=[berths, notices])


@pytest.mark.parametrize(
    ("text", "hints"),
    [
        # The real port run: the 7B model marked all three of these as "documents".
        ("Berth allocation. Find berths that fit a vessel's length and draft.", ["table_lookup"]),
        (
            "Weather forecast. Check the wind and weather forecast for the arrival window.",
            ["weather"],
        ),
        (
            "Port charges estimation. Estimate berth hire, port dues and pilotage.",
            ["arithmetic", "table_lookup"],
        ),
        # One generic shared word ("port") must not drag in a table lookup.
        ("Port rules and notices. Answer questions about notices to port users.", []),
        (
            "Pre-arrival deadline. How many hours before arrival is today's deadline?",
            ["current_date"],
        ),
    ],
)
def test_infer_tool_hints(text: str, hints: list[str]) -> None:
    from forgeai.tools.registry import infer_tool_hints

    assert sorted(infer_tool_hints(text, _port_corpus())) == sorted(hints)


async def test_weather_country_disambiguates_and_reports_misses() -> None:
    """Real port run: 'Kochi' alone resolved to Kochi, Japan (a 9-20 °C forecast)."""
    async with httpx.AsyncClient(transport=weather_transport([])) as client:
        bare = await call_tool(
            REGISTRY["weather"], {"location": "Kochi", "days": 1}, env(http=client)
        )
        wrong = await call_tool(REGISTRY["weather"], {"location": "Kochi, Peru"}, env(http=client))
        by_code = await call_tool(
            REGISTRY["weather"], {"location": "Kochi, IN", "days": 1}, env(http=client)
        )
    assert "Kochi, Japan" in bare.output  # no country: first match, and it says which
    assert "place not found: 'Kochi' in Peru" in wrong.output
    assert "Kochi, India" in by_code.output
