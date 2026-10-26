# What I wrote

This assignment came with a starter repository from the course staff: the folder layout, function and class signatures with docstrings, some helper code and an automatic test suite. My job was to implement the missing parts until the tests passed.

- Starter code: [`80d46db`](https://github.com/iqueipopg/icai-imat/commit/80d46db24273a2ffad75aaf40806cb828fcd527e) (2025-11-06), as distributed through GitHub Classroom.
- My work:
  - [`d1bb733`](https://github.com/iqueipopg/icai-imat/commit/d1bb733c2d2f79eea214f73698fa70395dd4c314) (2025-11-26): implement DCGAN generator, discriminator and training loop

The diff between the two is exactly my contribution:

```bash
git diff 80d46db d1bb733 -- computer-vision-ii/dcgan-mnist
```

| File | Lines added | Lines removed |
|---|---|---|
| `src/model.py` | 72 | 1 |
| `src/train.py` | 78 | 19 |

All 5 tests of the course suite pass. I reran them on 2026-10-26 against the code in this folder.

Not included here: the assignment statement, the course test suite, datasets and trained weights.
