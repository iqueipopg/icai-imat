# What I wrote

This assignment came with a starter repository from the course staff: the folder layout, function and class signatures with docstrings, some helper code and an automatic test suite. My job was to implement the missing parts until the tests passed.

- Starter code: [`6c31e42`](https://github.com/iqueipopg/icai-imat/commit/6c31e42a337ae1fcb9479c27ed245fc31af48a77) (2025-03-25), as distributed through GitHub Classroom.
- My work:
  - [`9eff971`](https://github.com/iqueipopg/icai-imat/commit/9eff971644430fd1f28e1aaaa9f332cc9800823f) (2025-04-02): implement RNN forward and backward, dataset and training loop
  - [`2e84348`](https://github.com/iqueipopg/icai-imat/commit/2e84348062bbb3684479e7fea696ff46cfe72fcd) (2025-04-02): fix lint and type checks

The diff between the two is exactly my contribution:

```bash
git diff 6c31e42 2e84348 -- deep-learning/rnn-electricity-price
```

| File | Lines added | Lines removed |
|---|---|---|
| `src/data.py` | 60 | 3 |
| `src/evaluate.py` | 37 | 17 |
| `src/models.py` | 86 | 0 |
| `src/train.py` | 69 | 3 |
| `src/train_functions.py` | 80 | 1 |
| `src/utils.py` | 1 | 0 |

4 tests of the course suite pass (the performance tests, which need the trained weights, were not rerun). I reran them on 2026-10-26 against the code in this folder.

Not included here: the assignment statement, the course test suite, datasets and trained weights.
