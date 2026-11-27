# Scorecard: agent v0 (holdout)

## holdout: 6/8 passed (75%)

- refusal accuracy 88%, groundedness 100%, fact recall 92%
- by category: ambiguous 100%, answerable 100%, multi_doc 0%, out_of_scope 50%
- failures: hallucination 1, wrong_refusal 1

Judge canary accuracy 100% (trusted).

| Case | Split | Category | Result | Failure | Reason |
|---|---|---|---|---|---|
| c05 | holdout | answerable | pass |  |  |
| c08 | holdout | answerable | pass |  |  |
| c09 | holdout | answerable | pass |  |  |
| c11 | holdout | answerable | pass |  |  |
| c15 | holdout | ambiguous | pass |  |  |
| c16 | holdout | multi_doc | **fail** | hallucination | facts not stated correctly: within 10 working days |
| c20 | holdout | out_of_scope | **fail** | wrong_refusal | answered a question it must refuse |
| c21 | holdout | out_of_scope | pass |  |  |
