TASK: intake.requirements

Write the requirements for an AI assistant from the brief, the document summary and the
requester's answers. The answers override your own guesses.

Rules:
- `agent_name`: short and descriptive, e.g. "Shore Staff HR Desk".
- `must_answer`: 3 to 12 specific topics the assistant must handle, one topic each.
  Set `answered_from` to "documents" when the answer should come from the documents.
  Set it to "tool" only when the topic needs live data or computation, and set
  `tool_hint` to one of: table_lookup (finding rows in a CSV table), arithmetic,
  weather, current_date.
- `must_refuse`: at least 2 topics the assistant must decline (outside scope, personal
  or confidential data, advice it must not give), each with a reason.
- Only list must_answer topics that the documents plausibly cover or a tool can serve.
  Every documents topic is checked against the documents in the next step.

BRIEF:
$brief

DOCUMENTS:
$corpus_summary

REQUESTER ANSWERS:
$answers
