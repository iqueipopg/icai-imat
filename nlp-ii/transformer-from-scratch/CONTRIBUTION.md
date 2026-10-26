# What I wrote

This assignment came with a starter repository from the course staff: the folder layout, function and class signatures with docstrings, some helper code and an automatic test suite. My job was to implement the missing parts until the tests passed.

- Starter code: [`131884f`](https://github.com/iqueipopg/icai-imat/commit/131884fe65dce77a7abf5ae0f501827c24a5cb8c) (2025-09-25), as distributed through GitHub Classroom.
- My work:
  - [`3dda88e`](https://github.com/iqueipopg/icai-imat/commit/3dda88e46c99577c5645e63e41c28e494cd0a7ba) (2025-10-02): implement attention heads, encoder, decoder and transformer

The diff between the two is exactly my contribution:

```bash
git diff 131884f 3dda88e -- nlp-ii/transformer-from-scratch
```

| File | Lines added | Lines removed |
|---|---|---|
| `src/decoder.py` | 17 | 15 |
| `src/encoder.py` | 12 | 11 |
| `src/transformer.py` | 89 | 86 |
| `src/utils.py` | 20 | 18 |

All 23 tests of the course suite pass. I reran them on 2026-10-26 against the code in this folder.

Not included here: the assignment statement, the course test suite, datasets and trained weights.
