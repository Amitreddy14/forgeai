# Change log

## Iteration 1: add refuse topic: "Real-time operational queries"

- category: wrong_refusal (proposed by critic)
- targets: c26, c27
- why: Involves specific vessel or port operations that should be handled by the relevant stakeholders.
- dev pass rate: 74% -> 68%
- fixed: none; broke: c24
- decision: reverted: 0 fixed, 1 broken (no net gain)
- tokens: 32,496

## Iteration 1: add rule: "Refuse questions that require personal information or violate privacy policies, and direct users to the appropriate department for such requests."

- category: wrong_refusal (proposed by critic)
- targets: c26, c27
- why: This rule ensures privacy is maintained and directs users to the correct department for sensitive information.
- dev pass rate: 74% -> 74%
- fixed: c27; broke: c24
- decision: reverted: 1 fixed, 1 broken (no net gain)
- tokens: 31,491

## Iteration 1: add rule: "Extract and verify all relevant facts from the retrieved passage before providing an answer."

- category: prompt_defect (proposed by critic)
- targets: c14, c21
- why: Ensures all pertinent details are captured and accurately conveyed, preventing omissions or misstatements.
- dev pass rate: 74% -> 68%
- fixed: none; broke: c24
- decision: reverted: 0 fixed, 1 broken (no net gain)
- tokens: 32,744
