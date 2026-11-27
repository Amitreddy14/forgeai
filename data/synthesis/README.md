# How the demo documents were made

Both demo document sets are **LLM-synthesized**. They describe a fictional company
(Konkan Marine Services) and invented port tariffs and notices. Every document says on
its first line that it is synthetic. Do not use them for real HR or navigation decisions.

Pipeline:

1. An LLM (Claude, Anthropic) was given the prompt in `hr-shore-staff.md` or
   `port-call-ops.md` and wrote each document as Markdown, saved under
   `data/demos/<run>/source/`. The berth register CSV was written by the same prompt.
2. A person read every document for internal consistency: numbers in the tariff match
   the berth register, and the notices do not contradict the tariff.
3. `scripts/render_demo_docs.py` renders the Markdown sources to the PDFs in
   `data/demos/<run>/docs/` (reportlab, deterministic). The factory only ever reads
   the PDFs and the CSV, never the Markdown.

The briefs are written by hand. Each one plants a deliberate trap:

- **hr-shore-staff** asks the agent to tell employees how many leave days they have
  left. That needs each employee's records, which the documents do not contain. The
  groundability guard should flag it and the agent should refuse it.
- **port-call-ops** needs a table lookup, arithmetic and a live weather forecast, but
  never asks for today's date. The factory should attach `csv_lookup`, `calculator`
  and `weather`, and not `current_date`.
