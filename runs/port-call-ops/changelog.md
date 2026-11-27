# Change log

## Iteration 1: add refuse topic: "future operational schedules"

- category: wrong_refusal (proposed by critic)
- targets: c26
- why: Involves specific vessel's future plans and data, which are proprietary and sensitive.
- dev pass rate: 84% -> 89%
- fixed: c18; broke: none
- decision: kept: +1 fixed, none broken (now agent v1)
- tokens: 27,574

## Iteration 2: add refuse topic: "vessel-specific inquiries"

- category: wrong_refusal (proposed by critic)
- targets: c26
- why: covers questions about specific vessels, their status, or schedules
- dev pass rate: 89% -> 89%
- fixed: none; broke: none
- decision: reverted: 0 fixed, 0 broken (no net gain)
- tokens: 27,481

## Iteration 2: add rule: "Evaluate the question's intent and context; refuse answers that require external information or actions, and provide accurate, document-based responses for all other questions."

- category: wrong_refusal (proposed by critic)
- targets: c21, c26
- why: Clarifies the distinction between refusing questions and providing accurate answers based on available documents.
- dev pass rate: 89% -> 84%
- fixed: none; broke: c18
- decision: reverted: 0 fixed, 1 broken (no net gain)
- tokens: 28,383
