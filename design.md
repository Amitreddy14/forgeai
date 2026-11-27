# ForgeAI — Design

## Subject, audience, job

- **Subject:** a factory that builds an AI agent and then *surveys* it before it is
  allowed to sail. OceanAI is a maritime company, and shipping already has a precise,
  serious ritual for "prove this vessel is fit": the classification survey and its
  certificate, with *conditions of class* for anything not yet fixed.
- **Audience:** OceanAI engineers at the review, and platform admins after it.
- **Primary job:** watch a build happen, then inspect the evidence — what was asked,
  what was built, what was tested, what still fails.

## Concept: the survey report

The agent is the vessel. A run is a survey. The final screen is a **certificate of
survey**: the agent's name and particulars, its holdout score drawn as a **load line
(Plimsoll mark)**, and a numbered list of **conditions** — the open failures, each
with its cause. Requirements read like the vessel's particulars; the changelog reads
like a survey log with entries kept or struck through.

The metaphor lives in structure and vocabulary, not in decoration: no anchors, waves,
rope borders or ship clip-art.

## The one bold element

**The load line.** The scorecard's headline number is drawn as a Plimsoll mark: a ring
crossed by a horizontal bar at the pass mark, with the score as a water level against
it. Dev score and holdout score are two marks on the same scale (like summer and
winter load lines). Everything else on the page stays quiet so this reads first.

## Colour

Drawn from Admiralty nautical charts: cool chart paper, deep-water blue, shallow-water
tint, and signal colours used only for state.

| Token | Hex | Use |
|---|---|---|
| `--paper` | `#EEF2F3` | page background (cool chart white, not cream) |
| `--sheet` | `#FFFFFF` | certificate sheet, panels |
| `--ink` | `#0F2A3F` | text, rules, the load line (deep sounding blue) |
| `--shoal` | `#BFD6E2` | shallow-water tint: selected rows, score fill |
| `--signal-pass` | `#2E7D5B` | pass |
| `--signal-fail` | `#B3261E` | fail, conditions |
| `--signal-hold` | `#E0A800` | flagged / assumed / reverted patch |

Pass/fail never rely on colour alone: every status also has a word and a glyph.

## Type

- **Archivo** (variable width): UI and headings. Headings use the expanded width at
  weight 600 — the stencilled, wide look of hull markings and certificate titles.
  Body at normal width, 16 px / 1.55.
- **Source Serif 4**: long-form report prose (`report.md` rendered), 17 px / 1.65,
  max 68 characters per line.
- **JetBrains Mono**: only for things that *are* code — prompts, JSON, diffs, chunk text.
  Never for labels.

Scale (1.25): 13 · 16 · 20 · 25 · 31 · 39. Sentence case everywhere. No all-caps labels.

## Layout

```
┌──────────────────────────────────────────────────────────────┐
│ ForgeAI                     Builds   New build   Chat        │  plain top bar
├───────────────┬──────────────────────────────────────────────┤
│ Survey log    │  Certificate of survey                       │
│ (stage list,  │  Agent: HR policy desk · v3                  │
│  done/active/ │  ┌────────┐  Holdout 78%   pass mark 70%      │
│  failed)      │  │ load   │  Dev 84%                         │
│               │  │ line   │                                  │
│               │  └────────┘                                  │
│               │  Conditions (open failures)                  │
│               │  1. Two-document question … retrieval miss   │
│               │  Particulars | Evals | Changelog | Definition │
└───────────────┴──────────────────────────────────────────────┘
```

- Left-aligned throughout; a fixed left rail for the survey log (the stage timeline,
  which *is* a sequence, so it is numbered), content column max 960 px.
- Tables for evals and changelog — the data is tabular. No card grids.
- Radius: 2 px on inputs/buttons only; sheets are square. Hierarchy comes from the
  1 px ink rule and whitespace, not shadows.
- Mobile: rail collapses above the content as a horizontal stage strip.

## Motion

One moment only: while a build runs, the current stage in the survey log fills like a
tide gauge (CSS height transition driven by HTMX progress). Everything else is static.
`prefers-reduced-motion` turns it into a plain state change.

## Words

Plain verbs, user's vocabulary: "Start build", "Answer questions", "Approve
requirements", "Chat with this agent", "Download definition". A button and its result
share the verb ("Approve requirements" → "Requirements approved"). Errors say what
happened and what to do ("The model server did not answer. Check FORGE_LLM_BASE_URL and
try again."). Empty state on Builds: "No builds yet. Start one from a brief and a folder
of documents."

## Review against generic defaults (what I changed)

- First draft used a dark "control room" theme with a green accent → that is the
  near-black + acid accent default. Replaced with chart paper + deep blue.
- First draft used a big percentage with a small label as the hero → the default stat
  treatment. Replaced with the load-line drawing, which only makes sense for this subject.
- Dropped a monospace face for status labels; mono is reserved for real code.
- Avoided cards for eval cases; they are rows in a table because reviewers compare them.

## Accessibility floor

WCAG AA contrast (ink on paper ≈ 13:1), visible 2 px focus ring in `--ink`, all
interactive elements keyboard-reachable, `aria-live="polite"` on build progress,
responsive to 360 px.
