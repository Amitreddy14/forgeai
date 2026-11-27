TASK: intake.requirements

Write the requirements for an AI assistant from the brief, the document summary and the
requester's answers. The answers override your own guesses.

Rules:
- `agent_name`: short and descriptive, e.g. "Shore Staff HR Desk".
- `must_answer`: 3 to 12 specific topics, ONE ask per topic. Cover every ask in the brief.
  - `answered_from`: "documents" when the answer is written in the documents; "tool" when
    it needs live data or computation.
  - `tool_hints`: for tool topics, any of: table_lookup (finding rows in a CSV table),
    arithmetic (estimates, totals, costs), weather (forecasts, wind), current_date.
  - If the brief asks for something about an individual person (their balance, history,
    record, salary), make that its own topic. Never merge it into a general policy topic.
- `must_refuse`: at least 2 topics the assistant must decline (outside scope, personal
  or confidential data, advice it must not give), each with a reason.
- Every documents topic is checked against the documents in the next step, so list
  what the brief asks for, even if you doubt the documents cover it.

Example topics (shape only):
{"topic": "Annual leave entitlement", "detail": "days earned per year", "answered_from": "documents", "tool_hints": []}
{"topic": "Remaining leave balance", "detail": "days an employee has left this year", "answered_from": "documents", "tool_hints": []}
{"topic": "Port charge estimate", "detail": "total of dues and pilotage for a vessel", "answered_from": "tool", "tool_hints": ["arithmetic"]}

BRIEF:
$brief

DOCUMENTS:
$corpus_summary

REQUESTER ANSWERS:
$answers
