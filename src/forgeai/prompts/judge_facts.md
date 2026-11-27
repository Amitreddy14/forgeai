TASK: judge.facts

You grade one answer against a list of required facts. Judge only meaning, not style.

QUESTION:
$question

REQUIRED FACTS:
$facts

ANSWER TO GRADE:
$answer

For each required fact, in the order given, return:
- `status`:
  - "present": the answer states this fact (wording may differ; every number must be equal).
  - "contradicted": the answer states something incompatible (a different number, the
    opposite rule).
  - "missing": the answer does not state it.
- `evidence`: for "present" or "contradicted", the exact words from the ANSWER that state
  it, copied character for character. For "missing", an empty string.

Your evidence is checked against the answer. A long or confident answer is not better:
if you cannot copy words from the answer that state the fact, it is missing.
