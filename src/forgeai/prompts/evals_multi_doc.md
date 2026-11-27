TASK: evals.multi_doc

You write test questions that need two passages to answer. The assistant serves:
$audience.

PASSAGE A [$chunk_a] from "$title_a":
$passage_a

PASSAGE B [$chunk_b] from "$title_b":
$passage_b

Write ONE natural question whose complete answer needs a fact from passage A AND a
fact from passage B (for example comparing two limits, or combining two rules). If the
passages have nothing a user would sensibly combine, set `possible` to false.

Rules:
- `fact_a` / `fact_b`: a short fact (2 to 8 words) copied word for word from A / from B.
- `quote_a` / `quote_b`: the sentence containing that fact, copied exactly from A / B.
