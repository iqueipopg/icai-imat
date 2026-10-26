# What I wrote

This assignment came with a starter repository from the course staff: the folder layout, function and class signatures with docstrings, some helper code and an automatic test suite. My job was to implement the missing parts until the tests passed.

- Starter code: [`e90b6db`](https://github.com/iqueipopg/icai-imat/commit/e90b6dbbbe9f3f48f6722c0840a4ec7da280b619) (2025-03-11), as distributed through GitHub Classroom.
- My work:
  - [`e6293f0`](https://github.com/iqueipopg/icai-imat/commit/e6293f0ae621ff1f39d2dc7d8a7fd1c686997d46) (2025-03-20): implement SGD, momentum, Nesterov, Adam, StepLR and dropout; fine-tune ResNet-18

The diff between the two is exactly my contribution:

```bash
git diff e90b6db e6293f0 -- deep-learning/optimizers-from-scratch
```

| File | Lines added | Lines removed |
|---|---|---|
| `src/data.py` | 21 | 3 |
| `src/evaluate.py` | 76 | 19 |
| `src/models.py` | 87 | 1 |
| `src/optimization.py` | 272 | 1 |
| `src/train.py` | 173 | 2 |
| `src/utils.py` | 16 | 0 |

67 tests of the course suite pass (the performance tests, which need the trained weights, were not rerun). I reran them on 2026-10-26 against the code in this folder.

Not included here: the assignment statement, the course test suite, datasets and trained weights.
