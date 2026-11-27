# hr-mini

A tiny example for trying the factory on real models in a few minutes. The brief
deliberately asks for something the documents do not contain (salary bands), so the
groundability guard should flag it.

```bash
forge intake --brief data/examples/hr-mini/brief.md --docs data/examples/hr-mini/docs --answers auto --run-id scratch-hr-mini
forge generate --run-id scratch-hr-mini --docs data/examples/hr-mini/docs
forge chat runs/scratch-hr-mini/agent.v0.json
forge evalgen --run-id scratch-hr-mini --docs data/examples/hr-mini/docs --min-cases 12
forge evaluate --run-id scratch-hr-mini --docs data/examples/hr-mini/docs --split all
```

Two short documents cannot yield the 20 cases the full factory requires, hence
`--min-cases 12` here.

The generated HR agent should have no tools.

Use `--answers interactive` to answer the questions yourself, or
`--answers data/examples/hr-mini/answers.json` for scripted answers.
