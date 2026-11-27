TASK: evals.answerable

You write test questions for an assistant that answers from documents. The assistant
serves: $audience.

Write ONE question that a real user would ask and that this passage answers fully.
Then give the facts a correct answer must contain.

PASSAGE [$chunk_id] from "$doc_title":
$passage

Rules:
- The question must make sense on its own: name the subject (for example "sick leave"),
  never "this policy" or "the passage". It must not contain the answer.
- `key_facts`: 1 or 2 SHORT facts (2 to 8 words), copied word for word from the passage,
  that answer exactly what the question asks. No facts the question does not ask about.
- `quote`: the sentence(s) from the passage that contain all key facts, copied exactly.

Example of the shape (not of the content):
{"question": "How many days of annual leave do shore staff earn each year?",
 "key_facts": ["24 days of annual leave"],
 "quote": "Shore staff earn 24 days of annual leave per calendar year."}
