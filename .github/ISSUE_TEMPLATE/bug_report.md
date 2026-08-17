---
name: Bug report
about: Report unexpected behavior in the training pipeline
title: "[BUG] "
labels: bug
assignees: ""
---

## Describe the bug

A clear and concise description of what went wrong.

## Which pipeline step

- [ ] Step 1 — `transform` (`pipeline/transform.py`, `training_pairs`)
- [ ] Step 2 — `train` (`pipeline/train.py`, `model.joblib`)
- [ ] Step 3 — `index_search` (`pipeline/index_search.py`, `product_search`)
- [ ] Full pipeline (`python -m pipeline.main`)
- [ ] Tests (`pytest tests/`)
- [ ] Other / not sure

## Steps to reproduce

1. Run '...'
2. With environment variables '...'
3. See error

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened. Include the full traceback or log output if available.

```
paste log/traceback here
```

## Environment

- OS:
- Python version (`python --version`):
- Installed via: [venv / Docker]
- Relevant environment variables (redact secrets): `DATABASE_URL`, `REDIS_URL`,
  `ELASTICSEARCH_URL` set? (yes/no is enough — do not paste credentials)
- Upstream state: was `pocket-nutrition/community` reloaded and
  `pocket-nutrition/data` populated before running?

## Additional context

Any other context about the problem (recent config changes, data volume, etc.).
