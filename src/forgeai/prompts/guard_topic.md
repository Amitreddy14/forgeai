TASK: guard.topic

Decide whether the passages below contain the information needed to answer questions
about this topic.

TOPIC: $topic
DETAIL: $detail

PASSAGES:
$passages

Return:
- `status`: "supported" if the passages contain the needed information, "partial" if they
  cover only some of it (say what is missing in `note`), "unsupported" if they do not
  cover it.
- If supported or partial: `chunk_id` of the best passage and `quote`, 8 to 40 words
  copied exactly from that passage. Copy, do not paraphrase.
- If unsupported: leave `chunk_id` and `quote` empty.
