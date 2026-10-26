# What I wrote

This assignment came with a starter repository from the course staff: the folder layout, function and class signatures with docstrings, some helper code and an automatic test suite. My job was to implement the missing parts until the tests passed.

- Starter code: [`498ed28`](https://github.com/iqueipopg/icai-imat/commit/498ed2820f83fc4dc5b369eabda675ce884c8be1) (2025-10-02), as distributed through GitHub Classroom.
- My work:
  - [`73a2a62`](https://github.com/iqueipopg/icai-imat/commit/73a2a625b292e257383cbeeeac45f2d38cd2ecc9) (2025-10-21): implement saliency maps, occlusion, guided backprop and integrated gradients

The diff between the two is exactly my contribution:

```bash
git diff 498ed28 73a2a62 -- computer-vision-ii/explainability
```

| File | Lines added | Lines removed |
|---|---|---|
| `src/explainers/guided_backprop.py` | 48 | 10 |
| `src/explainers/integrated_gradients.py` | 42 | 0 |
| `src/explainers/occlusion.py` | 56 | 0 |
| `src/utils.py` | 28 | 0 |
| `src/visualize.py` | 52 | 0 |

All 37 tests of the course suite pass. I reran them on 2026-10-26 against the code in this folder.

Not included here: the assignment statement, the course test suite, datasets and trained weights.
