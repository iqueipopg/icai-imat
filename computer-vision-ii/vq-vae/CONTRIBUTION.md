# What I wrote

This assignment came with a starter repository from the course staff: the folder layout, function and class signatures with docstrings, some helper code and an automatic test suite. My job was to implement the missing parts until the tests passed.

- Starter code: [`442f3c8`](https://github.com/iqueipopg/icai-imat/commit/442f3c8a5e7f39839f2b1c133dbca3ee19d4d9e1) (2025-11-20), as distributed through GitHub Classroom.
- My work:
  - [`65d436a`](https://github.com/iqueipopg/icai-imat/commit/65d436a364113513cf124ec7877165e085f7be61) (2025-11-27): implement encoder, decoder, vector quantizer with EMA and losses

The diff between the two is exactly my contribution:

```bash
git diff 442f3c8 65d436a -- computer-vision-ii/vq-vae
```

| File | Lines added | Lines removed |
|---|---|---|
| `src/checkpoint_utils.py` | 1 | 1 |
| `src/config.py` | 1 | 1 |
| `src/ema_update.py` | 28 | 8 |
| `src/loss.py` | 16 | 40 |
| `src/networks.py` | 95 | 38 |
| `src/train.py` | 112 | 15 |
| `src/training.py` | 84 | 15 |
| `src/utils.py` | 1 | 1 |
| `src/vq_vae.py` | 142 | 26 |

All 38 tests of the course suite pass. I reran them on 2026-10-26 against the code in this folder.

Not included here: the assignment statement, the course test suite, datasets and trained weights.
