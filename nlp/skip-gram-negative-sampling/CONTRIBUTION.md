# What I wrote

This assignment came with a starter repository from the course staff: the folder layout, function and class signatures with docstrings, some helper code and an automatic test suite. My job was to implement the missing parts until the tests passed.

- Starter code: [`d344de9`](https://github.com/iqueipopg/icai-imat/commit/d344de946e77aed806de51f74f54001806620178) (2025-02-06), as distributed through GitHub Classroom.
- My work:
  - [`43a5dac`](https://github.com/iqueipopg/icai-imat/commit/43a5dac14cbcfea43be67635a0e3a8977deb18ae) (2025-02-21): implement subsampling, batching, skip-gram and negative sampling loss

The diff between the two is exactly my contribution:

```bash
git diff d344de9 43a5dac -- nlp/skip-gram-negative-sampling
```

| File | Lines added | Lines removed |
|---|---|---|
| `src/data_processing.py` | 81 | 26 |
| `src/main.py` | 39 | 10 |
| `src/skipgram.py` | 35 | 16 |
| `src/train.py` | 40 | 23 |

All 11 tests of the course suite pass. I reran them on 2026-10-26 against the code in this folder.

Not included here: the assignment statement, the course test suite, datasets and trained weights.
