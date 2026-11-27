TASK: critic.refuse_topic

An AI assistant answered questions it should have refused. Name ONE general category of
question it should refuse from now on, covering these cases and similar ones.

QUESTIONS IT SHOULD HAVE REFUSED:
$questions

TOPICS IT ALREADY REFUSES:
$topics

Rules:
- `topic`: 3 to 8 words naming a category, for example "a specific vessel's own data"
  or "future or planned policy changes". Not a copy of any question.
- It must not overlap the topics already refused.
Return `topic` and a one-sentence `rationale`.
