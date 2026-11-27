# Generated eval set

23 cases kept, 7 rejected by validation. Dev 15 / holdout 8.

Categories: ambiguous 3, answerable 12, multi_doc 3, out_of_scope 5

| ID | Split | Category | Question | Expected |
|---|---|---|---|---|
| c01 | dev | answerable | How long do employees have to declare new conflicts of interest? | within 30 days of arising |
| c02 | dev | answerable | How much bereavement leave do employees receive for the death of a sibling? | 5 working days |
| c03 | dev | answerable | What expenses are non-reimbursable for trips shorter than 4 nights? | alcohol |
| c04 | dev | answerable | How many times can an employee be late in a month before it counts as half a day of annual leave? | three times |
| c05 | holdout | answerable | How many days of annual leave do shore staff earn per year? | 24 working days of paid annual leave |
| c06 | dev | answerable | How much is reimbursed for using a personal two-wheeler for local transport? | INR 6 per kilometre |
| c07 | dev | answerable | How many days per week can remote work be approved for after 6 months? | 2 days per week |
| c08 | holdout | answerable | For what class of flights can business class be booked? | flights longer than 6 hours |
| c09 | holdout | answerable | How much paid leave do birth mothers receive for their first two children? | 26 weeks of paid maternity leave |
| c10 | dev | answerable | Within how many days must expense claims be submitted? | within 30 days of returning from the trip |
| c11 | holdout | answerable | How many unused annual leave days can be carried forward? | Up to 10 unused annual leave days |
| c12 | dev | answerable | What percentage of the estimated trip cost can employees request for a travel advance? | 75 percent of the estimated trip cost |
| c13 | dev | ambiguous | What's the max gift value from a partner in a yr? | INR 3,000 |
| c14 | dev | ambiguous | How many days in advance do you need to apply for annual leave? | 10 working days |
| c15 | holdout | ambiguous | What's the daily allowance for domestic travel in India? | INR 1,800 per day |
| c16 | holdout | multi_doc | How many days of paid sick leave do employees receive per year? Also, how long does the Head of HR have to hold a meeting after receiving an escalated grievance? | 12 days of paid sick leave per calendar year; within 10 working days |
| c17 | dev | multi_doc | How long does the Internal Committee have to complete its inquiry into a sexual harassment complaint? Also, how many paid public holidays does the company observe each year? | within 90 days; 12 paid public holidays each year |
| c18 | dev | multi_doc | How long do employees have to declare new conflicts of interest? Also, within how many days must expense claims be submitted? | within 30 days of arising; within 30 days of returning from the trip |
| c19 | dev | out_of_scope | Can you tell me how much my colleague earns? | refuse |
| c20 | holdout | out_of_scope | I want to know the status of my leave balance this year. | refuse |
| c21 | holdout | out_of_scope | How many days has John taken off this year? | refuse |
| c22 | dev | out_of_scope | Can you provide a list of all shore staff who have taken leave this month? | refuse |
| c23 | dev | out_of_scope | What are the plans for travel advances in the next quarter? | refuse |

## Rejected candidates

| Category | Question | Reason |
|---|---|---|
| answerable | How many days of leave without pay can shore staff request in a year? | no short, relevant key fact |
| answerable | What does this code set out? | no short, relevant key fact |
| answerable | What disciplinary actions can be taken for serious misconduct? | key fact 'a written warning' is not in the quoted source |
| answerable | Which staff are covered by this policy for travel expenses? | question depends on unseen context (e.g. 'this policy') |
| out_of_scope | What are the HR policies for remote work? | answerable from the documents (names the must-answer topic "Remote work policy") |
| out_of_scope | How do we handle gifts and hospitality for our clients? | answerable from the documents (names the must-answer topic "Gifts and hospitality policy") |
| multi_doc | How many times can an employee be late in a month before it counts as half a day of annual leave? Also, how many unused annual leave days can be carried forward? | near-duplicate of another question |
