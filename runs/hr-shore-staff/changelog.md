# Change log

## Iteration 1: add refuse topic: "future financial forecasts"

- category: wrong_refusal (proposed by critic)
- targets: c22, c23
- why: Involves predictive financial data that could impact budgeting and financial planning processes.
- dev pass rate: 87% -> 93%
- fixed: c22; broke: none
- decision: kept: +1 fixed, none broken (now agent v1)
- tokens: 18,165

## Iteration 2: add refuse topic: "company-specific financial strategies"

- category: wrong_refusal (proposed by critic)
- targets: c23
- why: Involves detailed financial planning and could include future projections or sensitive data.
- dev pass rate: 93% -> 87%
- fixed: none; broke: c22
- decision: reverted: 0 fixed, 1 broken (no net gain)
- tokens: 18,235

## Iteration 2: add rule: "Evaluate the question's context and ensure it aligns with the document's scope before answering or refusing."

- category: wrong_refusal (proposed by critic)
- targets: c23
- why: This rule ensures the AI considers the question's context, preventing both incorrect answers and unjustified refusals.
- dev pass rate: 93% -> 87%
- fixed: none; broke: c22
- decision: reverted: 0 fixed, 1 broken (no net gain)
- tokens: 18,453
