# Demo runs

Two deliberately different briefs and document sets. Both were run end to end with
`forge build` (unattended, `--answers auto`) on local Ollama models
(`qwen2.5-8k` generator, `llama3.1-8k` judge). Every artifact is committed under
`runs/<run-id>/`.

| | hr-shore-staff | port-call-ops |
|---|---|---|
| Documents | 3 policy PDFs (leave, travel and expenses, code of conduct) | berth register CSV + tariff PDF + port notices PDF |
| Planted trap | brief asks for each employee's remaining leave days (not in the documents) | brief needs table lookup, arithmetic and weather, but not today's date |
| Expected factory behaviour | flag the gap, refuse it, attach no tools | attach csv_lookup, calculator, weather; not current_date |

Reproduce (the PDFs are rendered from `source/` by `scripts/render_demo_docs.py`):

```bash
forge build --brief data/demos/hr-shore-staff/brief.md --docs data/demos/hr-shore-staff/docs --run-id hr-shore-staff
forge build --brief data/demos/port-call-ops/brief.md --docs data/demos/port-call-ops/docs --run-id port-call-ops
```

How the documents were made: [data/synthesis/README.md](../synthesis/README.md).
