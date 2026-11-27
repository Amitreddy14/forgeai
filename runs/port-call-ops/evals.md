# Generated eval set

28 cases kept, 3 rejected by validation. Dev 19 / holdout 9.

Categories: ambiguous 3, answerable 12, multi_doc 4, out_of_scope 6, tool 3

| ID | Split | Category | Question | Expected |
|---|---|---|---|---|
| c01 | holdout | answerable | What is the maximum length of a vessel that can berth at berth KOC-CT1 in Kochi? | max_loa_m: 350 |
| c02 | dev | answerable | What is the requirement for visitors to a vessel? | need a port entry pass |
| c03 | dev | answerable | How many tugs are required for vessels longer than 250 meters? | 3 tugs |
| c04 | dev | answerable | How far south of Prongs Reef light does the pilot board at Mumbai? | 2 nautical miles south |
| c05 | holdout | answerable | What surcharge applies to pilotage and towage performed between 22:00 and 06:00 or on a public holiday? | 25 percent |
| c06 | dev | answerable | What percentage of the foreign-going rate do coastal vessels pay for port dues and pilotage? | 60 percent of the foreign-going rate |
| c07 | dev | answerable | How many hours before arrival must the pre-arrival notification be submitted? | at least 72 hours before the estimated time of arrival |
| c08 | holdout | answerable | What is the minimum pilotage fee per movement? | USD 1,200 per movement |
| c09 | dev | answerable | Where must vessels carrying dangerous goods anchor? | in the separate dangerous goods anchorage |
| c10 | dev | answerable | How long is this tariff schedule valid from? | Valid from 1 April 2026 to 31 March 2027 |
| c11 | holdout | answerable | What percentage of port dues must vessels calling only for bunkering, crew change or stores pay? | 50 percent of port dues |
| c12 | dev | answerable | What are the restrictions on the Kochi approach channel during the monsoon season? | vessels with a draft of no more than 12.5 metres |
| c13 | holdout | ambiguous | What's the rule for bunkering at anchorage? | allowed in daylight hours only |
| c14 | dev | ambiguous | What must the ship's agent do before docking? | deposit 100 percent of the estimated port charges |
| c15 | dev | ambiguous | What time can ships over 250 meters enter Kochi channel? | between 06:00 and 18:00 |
| c16 | dev | multi_doc | What rebates can vessels receive for using shore power? Also, what is the fixed waste fee per vessel call? | an additional 5 percent rebate on berth hire; USD 350 per call |
| c17 | dev | multi_doc | How many tugs are required for vessels longer than 250 meters? Also, where must vessels carrying dangerous goods anchor? | 3 tugs; in the separate dangerous goods anchorage |
| c18 | dev | multi_doc | What is the requirement for visitors to a vessel? Also, what percentage of port dues must vessels calling only for bunkering, crew change or stores pay? | need a port entry pass; 50 percent of port dues |
| c19 | holdout | multi_doc | How many hours before arrival must the pre-arrival notification be submitted? Also, how long is this tariff schedule valid from? | at least 72 hours before the estimated time of arrival; Valid from 1 April 2026 to 31 March 2027 |
| c20 | dev | tool | What is the max loa m for berth id KOC-CR1? | 300 (via csv_lookup) |
| c21 | dev | tool | What is the total daily berth hire usd for 3 days at berth id KOC-CR1? | 13500 (via calculator) |
| c22 | holdout | tool | What is the weather forecast for Kochi for the next two days? | call weather |
| c23 | holdout | out_of_scope | Can you provide customs clearance advice for our vessel? | refuse |
| c24 | dev | out_of_scope | Do you have any information on the personal schedule of the port manager? | refuse |
| c25 | holdout | out_of_scope | Could you tell me about the recent changes in port rules? | refuse |
| c26 | dev | out_of_scope | Is there a specific berth available for our vessel next week? | refuse |
| c27 | dev | out_of_scope | Can you share the personal contact details of the port agent handling our vessel? | refuse |
| c28 | dev | out_of_scope | What is the status of my colleague's application for a security clearance? | refuse |

## Rejected candidates

| Category | Question | Reason |
|---|---|---|
| answerable | How much of the estimated port charges must be deposited before berthing? | quote is not a verbatim span of the source passage |
| answerable | What is the maximum length of a vessel that can berth at this berth in meters? | near-duplicate of another question |
| out_of_scope | What are the current port charges for our vessel? | answerable from the documents (similarity 0.78 to the documents is at or above the answerable-question floor 0.78) |
