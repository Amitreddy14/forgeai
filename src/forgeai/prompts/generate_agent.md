TASK: generate.agent

You design an AI assistant that answers questions from a set of documents. Write its
role prompt and behaviour rules from the requirements below, and choose its tools.

REQUIREMENTS:
$requirements

DOCUMENTS:
$corpus_summary

AVAILABLE TOOLS:
$tools

Return:
- `role_prompt`: 3 to 6 sentences in the second person ("You are ..."). Say who the
  assistant serves, what it covers, and that it answers only from the provided
  documents. Do not list the refusal topics here; they are added separately.
- `rules`: 2 to 6 short, specific behaviour rules for this domain (for example how to
  phrase amounts, which section names to mention, when to ask the user to contact a
  person). No generic rules about citations or refusing; those are already enforced.
- `refusal_message`: one polite sentence the assistant says when it must decline,
  pointing the user to the right human contact if the requirements name one.
- `tools`: only the tools some must-answer topic cannot be answered without, each with a
  one-sentence `justification` naming that topic. An empty list is a valid answer.
