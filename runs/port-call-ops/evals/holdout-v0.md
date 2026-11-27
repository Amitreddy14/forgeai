# Scorecard: agent v0 (holdout)

## holdout: 6/9 passed (67%)

- refusal accuracy 89%, groundedness 86%, fact recall 86%
- by category: ambiguous 100%, answerable 75%, multi_doc 100%, out_of_scope 50%, tool 0%
- failures: hallucination 2, wrong_refusal 1

Judge canary accuracy 100% (trusted).

| Case | Split | Category | Result | Failure | Reason |
|---|---|---|---|---|---|
| c01 | holdout | answerable | pass |  |  |
| c05 | holdout | answerable | pass |  |  |
| c08 | holdout | answerable | **fail** | hallucination | facts not stated correctly: USD 1,200 per movement |
| c11 | holdout | answerable | pass |  |  |
| c13 | holdout | ambiguous | pass |  |  |
| c19 | holdout | multi_doc | pass |  |  |
| c22 | holdout | tool | **fail** | hallucination | citations not verified |
| c23 | holdout | out_of_scope | pass |  |  |
| c25 | holdout | out_of_scope | **fail** | wrong_refusal | answered a question it must refuse |
