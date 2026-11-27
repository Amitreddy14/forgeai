TASK: evals.ambiguous

You write hard test questions for an assistant that answers from documents. The
assistant serves: $audience.

Write ONE question that this passage answers, but phrase it the way a hurried user
would: informal, vague or indirect wording, a synonym instead of the document's term,
or a small typo. It must still have one clear correct answer in the passage.

PASSAGE [$chunk_id] from "$doc_title":
$passage

Rules:
- The question must name its subject (never "this policy") and not contain the answer.
- `key_facts`: 1 or 2 SHORT facts (2 to 8 words) copied word for word from the passage,
  answering exactly what is asked.
- `quote`: the sentence(s) containing the key facts, copied exactly.
