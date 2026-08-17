---
name: Feature request
about: Suggest an improvement to the training pipeline
title: "[FEATURE] "
labels: enhancement
assignees: ""
---

## Problem

What problem would this feature solve? Is it related to model accuracy, training pair
coverage, search index quality, or something else?

## Proposed solution

Describe the change you'd like to see. If it affects a specific pipeline step, name it:

- [ ] Step 1 — `transform` (training pair generation, yield imputation)
- [ ] Step 2 — `train` (model, hyperparameter search, cross-validation)
- [ ] Step 3 — `index_search` (Elasticsearch document shape, OFF streaming, recipes)
- [ ] Something upstream (belongs in `pocket-nutrition/knowledge` or
  `pocket-nutrition/community` instead — see note below)

## Alternatives considered

Any alternative approaches you've thought about.

## Additional context

Note: if this request is really about adding more ingredients, mappings, or cooking
relationships, the right place is the `pocket-nutrition/knowledge` YAML catalogue (and a
`pocket-nutrition/community` reload) — this repository is a pure consumer of that data and
should not be patched to bootstrap or auto-generate knowledge entries.
