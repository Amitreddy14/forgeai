# Change log

## Iteration 1: add refuse topic: "Real-time operational queries"

- category: wrong_refusal (proposed by critic)
- targets: c26, c27
- why: Involves specific vessel or port operations that should be handled by the relevant stakeholders.
- dev pass rate: 79% -> 79%
- fixed: c27; broke: c06
- decision: reverted: 1 fixed, 1 broken (no net gain)
- tokens: 30,261

## Iteration 1: add rule: "Refuse questions that require personal information or violate privacy policies, and direct users to the appropriate department for such requests."

- category: wrong_refusal (proposed by critic)
- targets: c26, c27
- why: This rule ensures privacy is maintained and directs users to the correct department for sensitive information.
- dev pass rate: 79% -> 74%
- fixed: none; broke: c06
- decision: reverted: 0 fixed, 1 broken (no net gain)
- tokens: 31,546

## Iteration 1: add rule: "Ensure all numerical answers are directly supported by the document content."

- category: hallucination (proposed by critic)
- targets: c21
- why: This prevents the assistant from making up numbers not found in the documents.
- dev pass rate: 79% -> 74%
- fixed: none; broke: c24
- decision: reverted: 0 fixed, 1 broken (no net gain)
- tokens: 34,337
