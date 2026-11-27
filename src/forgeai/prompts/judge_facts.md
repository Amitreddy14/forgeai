TASK: judge.facts

You grade one answer against a list of required facts. Judge only meaning, not style.

QUESTION:
$question

REQUIRED FACTS:
$facts

ANSWER TO GRADE:
$answer

For each required fact, return its status:
- "present": the answer states this fact (wording may differ; numbers must be equal).
- "contradicted": the answer states something incompatible (a different number, the
  opposite rule).
- "missing": the answer does not state it.
Length earns nothing: a long or confident answer is not better. Extra information is
fine unless it contradicts a required fact. Return the facts in the order given.
