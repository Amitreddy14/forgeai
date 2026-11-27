# Change log

## Iteration 1: add refuse topic: "company's financial forecasts"

- category: wrong_refusal (proposed by critic)
- targets: c23
- why: Involves sensitive and proprietary information about the company's future financial plans.
- dev pass rate: 93% -> 93%
- fixed: none; broke: none
- decision: reverted: 0 fixed, 0 broken (no net gain)
- tokens: 18,173

## Iteration 1: add rule: "Evaluate the question against the document's content and context before answering or refusing."

- category: wrong_refusal (proposed by critic)
- targets: c23
- why: This ensures the assistant considers the specific details of each query, avoiding incorrect refusals or answers.
- dev pass rate: 93% -> 93%
- fixed: none; broke: none
- decision: reverted: 0 fixed, 0 broken (no net gain)
- tokens: 18,302
