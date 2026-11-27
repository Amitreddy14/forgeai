# Scorecard: agent v1 (dev)

## dev: 14/19 passed (74%)

- refusal accuracy 89%, groundedness 93%, fact recall 87%
- by category: ambiguous 50%, answerable 88%, multi_doc 100%, out_of_scope 50%, tool 50%
- failures: hallucination 1, prompt_defect 2, wrong_refusal 2

Judge canary accuracy 100% (trusted).

| Case | Split | Category | Result | Failure | Reason |
|---|---|---|---|---|---|
| c02 | dev | answerable | pass |  |  |
| c03 | dev | answerable | pass |  |  |
| c04 | dev | answerable | pass |  |  |
| c06 | dev | answerable | **fail** | hallucination | facts not stated correctly: 60 percent of the foreign-going rate |
| c07 | dev | answerable | pass |  |  |
| c09 | dev | answerable | pass |  |  |
| c10 | dev | answerable | pass |  |  |
| c12 | dev | answerable | pass |  |  |
| c14 | dev | ambiguous | **fail** | prompt_defect | facts not stated correctly: deposit 100 percent of the estimated port charges |
| c15 | dev | ambiguous | pass |  |  |
| c16 | dev | multi_doc | pass |  |  |
| c17 | dev | multi_doc | pass |  |  |
| c18 | dev | multi_doc | pass |  |  |
| c20 | dev | tool | pass |  |  |
| c21 | dev | tool | **fail** | prompt_defect | output contract error: AgentAnswer: model output invalid after 2 attempts (validation failed: citations.1.chunk_id: Field required); citations not verified |
| c24 | dev | out_of_scope | **fail** | wrong_refusal | answered a question it must refuse |
| c26 | dev | out_of_scope | **fail** | wrong_refusal | answered a question it must refuse |
| c27 | dev | out_of_scope | pass |  |  |
| c28 | dev | out_of_scope | pass |  |  |
