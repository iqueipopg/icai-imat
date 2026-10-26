# What I wrote

This assignment came with a starter repository from the course staff: the folder layout, function and class signatures with docstrings, some helper code and an automatic test suite. My job was to implement the missing parts until the tests passed.

- Starter code: [`d465199`](https://github.com/iqueipopg/icai-imat/commit/d4651995ac7acb7954136216e1a3ebbdd2426f9d) (2025-09-19), as distributed through GitHub Classroom.
- My work:
  - [`f34d884`](https://github.com/iqueipopg/icai-imat/commit/f34d88494905e140cf34264b15e80a14c0b5aeb1) (2025-10-02): implement YOLO model, loss, dataset, NMS and trainer (mAP > 0.4 at epoch 40)

The diff between the two is exactly my contribution:

```bash
git diff d465199 f34d884 -- computer-vision-ii/yolo-from-scratch
```

| File | Lines added | Lines removed |
|---|---|---|
| `images/training_loss_0.png` | image | |
| `images/training_loss_100.png` | image | |
| `images/training_loss_50.png` | image | |
| `images/training_map.png` | image | |
| `src/constants.py` | 1 | 1 |
| `src/data/dataset.py` | 49 | 1 |
| `src/model/loss.py` | 73 | 3 |
| `src/model/model.py` | 44 | 8 |
| `src/model/train.py` | 116 | 15 |
| `src/utils.py` | 100 | 0 |

All 17 tests of the course suite pass. I reran them on 2026-10-26 against the code in this folder.

Not included here: the assignment statement, the course test suite, datasets and trained weights.
