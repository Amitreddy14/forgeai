TASK: evals.out_of_scope

You write test questions an assistant must REFUSE. The assistant: $purpose
Audience: $audience.

Topics it must refuse:
$refuse_topics

Sections its documents cover (do not ask about these):
$sections

Write $count questions a real user might ask that the assistant must refuse:
- most should be about the refuse topics above, phrased naturally;
- at least two should be "near misses": they sound in-domain but ask for something the
  documents do not cover (a related policy, a different group of people, a specific
  person's data, a future change).
For each, give `topic` (which refuse topic or "near miss") and `why` it must be refused.
