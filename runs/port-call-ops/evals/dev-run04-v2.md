# Scorecard: agent v2 (dev)

## dev: 16/19 passed (84%)

- refusal accuracy 89%, groundedness 100%, fact recall 97%
- by category: ambiguous 100%, answerable 100%, multi_doc 67%, out_of_scope 75%, tool 50%
- failures: prompt_defect 1, wrong_refusal 2

Judge canary accuracy 100% (trusted).

| Case | Split | Category | Result | Failure | Reason |
|---|---|---|---|---|---|
| c02 | dev | answerable | pass |  |  |
| c03 | dev | answerable | pass |  |  |
| c04 | dev | answerable | pass |  |  |
| c06 | dev | answerable | pass |  |  |
| c07 | dev | answerable | pass |  |  |
| c09 | dev | answerable | pass |  |  |
| c10 | dev | answerable | pass |  |  |
| c12 | dev | answerable | pass |  |  |
| c14 | dev | ambiguous | pass |  |  |
| c15 | dev | ambiguous | pass |  |  |
| c16 | dev | multi_doc | pass |  |  |
| c17 | dev | multi_doc | pass |  |  |
| c18 | dev | multi_doc | **fail** | prompt_defect | facts not stated correctly: need a port entry pass |
| c20 | dev | tool | pass |  |  |
| c21 | dev | tool | **fail** | wrong_refusal | refused an answerable question |
| c24 | dev | out_of_scope | pass |  |  |
| c26 | dev | out_of_scope | **fail** | wrong_refusal | answered a question it must refuse |
| c27 | dev | out_of_scope | pass |  |  |
| c28 | dev | out_of_scope | pass |  |  |
