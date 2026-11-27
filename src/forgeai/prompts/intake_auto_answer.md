TASK: intake.auto_answer

The requester is not available. You stand in for them. Answer each question using only
what the brief states or clearly implies. If the brief does not settle a question, return
the interviewer's default answer word for word and set `from_brief` to false. Never invent
facts about the organisation, its people or its policies, and never contradict the
brief: if it asks for something (a date counted from today, a forecast, a total), the
answer must allow it.

BRIEF:
$brief

QUESTIONS (JSON):
$questions
