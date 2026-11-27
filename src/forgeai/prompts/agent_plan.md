TASK: agent.plan

You can call tools before answering. Decide which calls are needed to answer the
question, or that none are needed.

TOOLS:
$tools

QUESTION:
$question

RELEVANT PASSAGES (first lines):
$passages

CALLS ALREADY MADE:
$history

Return `calls`: a list of {"name": ..., "args": {...}} to run now (at most 3), or an
empty list if you have everything you need. Do not repeat a call that already succeeded.
Use the calculator for any arithmetic. Use exact table ids and column names.
