# hr-mini

A tiny example for trying the factory on real models in a few minutes. The brief
deliberately asks for something the documents do not contain (salary bands), so the
groundability guard should flag it.

```bash
forge intake --brief data/examples/hr-mini/brief.md --docs data/examples/hr-mini/docs --answers auto --run-id scratch-hr-mini
```

Use `--answers interactive` to answer the questions yourself, or
`--answers data/examples/hr-mini/answers.json` for scripted answers.
