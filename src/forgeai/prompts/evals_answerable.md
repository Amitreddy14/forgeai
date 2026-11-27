TASK: evals.answerable

You write test questions for an assistant that answers from documents. The assistant
serves: $audience.

Write ONE question that a real user would ask and that this passage answers fully.
Then give the facts a correct answer must contain.

PASSAGE [$chunk_id] from "$doc_title":
$passage

Rules:
- The question must not contain the answer, and must make sense without the passage.
- `key_facts`: 1 to 3 short facts (2 to 8 words each, e.g. "24 days per year"), each
  copied word for word from the passage, numbers exactly as written.
- `quote`: the sentence(s) from the passage that contain all key facts, copied exactly.
