TASK: critic.rule

You improve an AI assistant that answers from documents. It failed some test cases.
Write ONE new behaviour rule that would fix this kind of failure in general.

FAILURE CATEGORY: $category
($category_help)

FAILED CASES:
$cases

CURRENT RULES:
$rules

Rules for your rule:
- One sentence, at most 35 words, about HOW to answer, not WHAT the answer is.
- It must help every question of this kind, not just these cases.
- Never put a specific answer, number, amount or name from the cases into the rule.
- Do not repeat a current rule.
Return `rule` and a one-sentence `rationale`.
