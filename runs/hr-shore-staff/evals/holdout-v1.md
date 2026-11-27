# Scorecard: agent v1 (holdout)

## holdout: 7/8 passed (88%)

- refusal accuracy 88%, groundedness 100%, fact recall 100%
- by category: ambiguous 100%, answerable 100%, multi_doc 100%, out_of_scope 50%
- failures: wrong_refusal 1

Judge canary accuracy 100% (trusted).

| Case | Split | Category | Result | Failure | Reason |
|---|---|---|---|---|---|
| c05 | holdout | answerable | pass |  |  |
| c08 | holdout | answerable | pass |  |  |
| c09 | holdout | answerable | pass |  |  |
| c11 | holdout | answerable | pass |  |  |
| c15 | holdout | ambiguous | pass |  |  |
| c16 | holdout | multi_doc | pass |  |  |
| c20 | holdout | out_of_scope | **fail** | wrong_refusal | answered a question it must refuse |
| c21 | holdout | out_of_scope | pass |  |  |
