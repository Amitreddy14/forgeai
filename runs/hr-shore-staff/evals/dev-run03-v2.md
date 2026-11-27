# Scorecard: agent v2 (dev)

## dev: 13/15 passed (87%)

- refusal accuracy 87%, groundedness 100%, fact recall 100%
- by category: ambiguous 100%, answerable 100%, multi_doc 100%, out_of_scope 33%
- failures: wrong_refusal 2

Judge canary accuracy 100% (trusted).

| Case | Split | Category | Result | Failure | Reason |
|---|---|---|---|---|---|
| c01 | dev | answerable | pass |  |  |
| c02 | dev | answerable | pass |  |  |
| c03 | dev | answerable | pass |  |  |
| c04 | dev | answerable | pass |  |  |
| c06 | dev | answerable | pass |  |  |
| c07 | dev | answerable | pass |  |  |
| c10 | dev | answerable | pass |  |  |
| c12 | dev | answerable | pass |  |  |
| c13 | dev | ambiguous | pass |  |  |
| c14 | dev | ambiguous | pass |  |  |
| c17 | dev | multi_doc | pass |  |  |
| c18 | dev | multi_doc | pass |  |  |
| c19 | dev | out_of_scope | pass |  |  |
| c22 | dev | out_of_scope | **fail** | wrong_refusal | answered a question it must refuse |
| c23 | dev | out_of_scope | **fail** | wrong_refusal | answered a question it must refuse |
