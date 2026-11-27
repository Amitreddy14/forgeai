# Synthesis prompt: port-call-ops

```text
Create a document set a ship agent would use to plan vessel calls at two Indian ports,
Kochi and Mumbai. All figures must be invented and the documents must say so on their
first line: they must never be mistaken for real tariffs or navigation notices.

1. A berth register as CSV with these columns:
   berth_id, port, terminal, berth_type, max_loa_m, max_draft_m, max_beam_m, max_dwt_t,
   cargo, daily_berth_hire_usd
   14 rows across both ports, covering general cargo, multipurpose, container, liquid
   bulk and cruise berths, with plausible size limits and daily rates.

2. "Port Tariff Schedule 2026": 8 numbered sections (applicability, port dues,
   pilotage, towage, berth hire, overtime and holidays, green vessel rebate, payment
   terms), each 60 to 100 words, with per-GT rates, minimums, surcharges and rebates
   in USD. Berth hire must refer to the daily rates in the berth register.

3. "Notices to Port Users 2026": 8 numbered sections (pre-arrival notification, pilot
   boarding, monsoon restrictions, night navigation, anchorage, bunkering, waste
   reception, security), each 60 to 100 words, with concrete limits (hours of notice,
   wind and wave limits in knots and metres, draft limits, fees).

Make the documents consistent with each other, and include limits that interact (for
example a monsoon draft limit that is lower than some berths' maximum draft), so that
good questions need two documents. Do not include customs, immigration or legal
procedures beyond one sentence; the assistant is meant to refuse those topics.
```
