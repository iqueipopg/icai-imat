# ICAI Mathematical Engineering and AI: coursework

Assignments from my degree in Mathematical Engineering and Artificial Intelligence at ICAI, Universidad Pontificia Comillas (2022-2026). Each one implements a method by hand: optimizers, backpropagation through time, attention, a detector, generative models.

Every assignment started from a starter repository from the course staff. Its folder has a `CONTRIBUTION.md` that links the starter commit and my commits, so the diff between them is my work and nothing else. The commits keep their original dates.

## Assignments

| Year | Course | Assignment | What it implements | Stack | Lines I wrote |
|---|---|---|---|---|---|
| 3 | Deep Learning | [Optimizers from scratch](deep-learning/optimizers-from-scratch) | SGD, momentum, Nesterov and Adam written as `torch.optim.Optimizer` subclasses, plus a step learning-rate scheduler and dropout, used to fine-tune a ResNet-18 on Imagenette. | PyTorch, torchvision, TensorBoard | 645 |
| 3 | Deep Learning | [RNN with a hand-written backward pass for day-ahead electricity prices](deep-learning/rnn-electricity-price) | Forward and backward pass of an RNN written as a custom `torch.autograd.Function`, and an LSTM that forecasts the 24 hourly day-ahead prices of the Nord Pool market. | PyTorch, pandas, TensorBoard | 333 |
| 3 | Natural Language Processing | [Skip-gram with negative sampling](nlp/skip-gram-negative-sampling) | Word2vec skip-gram trained with negative sampling on text8: subsampling of frequent words, context windows, the negative-sampling loss and a cosine-similarity check of the learned embeddings. | PyTorch, NumPy | 195 |
| 4 | Computer Vision II | [YOLO from scratch](computer-vision-ii/yolo-from-scratch) | Single-stage YOLO detector (7x7 grid, two boxes per cell) trained from scratch to detect raccoons, with its loss, IoU, non-maximum suppression and training loop. mAP above 0.4 at epoch 40. | PyTorch, torchmetrics | 383 |
| 4 | Computer Vision II | [Explainability for image classifiers](computer-vision-ii/explainability) | Saliency maps, occlusion, guided backpropagation and integrated gradients on a ResNet-50, plus a pixel-removal test of how faithful each explanation is. | PyTorch, torchvision, Matplotlib | 226 |
| 4 | Computer Vision II | [DCGAN on MNIST](computer-vision-ii/dcgan-mnist) | Deep convolutional GAN written from scratch: convolutional discriminator, transposed-convolution generator, their adversarial losses and the training loop on MNIST. | PyTorch, torchvision | 150 |
| 4 | Computer Vision II | [VQ-VAE](computer-vision-ii/vq-vae) | Vector-quantised variational autoencoder on CIFAR-10: residual encoder and decoder, a 512-entry codebook trained either with the codebook loss or with exponential moving averages. | PyTorch, torchvision | 480 |
| 4 | Natural Language Processing II | [Transformer from scratch](nlp-ii/transformer-from-scratch) | Encoder-decoder Transformer built from its parts: scaled dot-product attention heads, multi-head attention, feed-forward blocks, encoder and decoder layers with causal masking. | PyTorch | 138 |

## Course projects with their own repository

| Year | Course | Project | What it is |
|---|---|---|---|
| 1 | Programming | SnakeGame (private) | Snake in Python with new apple types that trigger events |
| 2 | Discrete Mathematics | [MadridGPS](https://github.com/iqueipopg/MadridGPS) | Shortest and fastest routes over the Madrid street graph: Dijkstra, Prim and Kruskal |
| 2 | Operating Systems Fundamentals | [BlockchainApp](https://github.com/iqueipopg/BlockchainApp) | Peer-to-peer blockchain with proof of work and a Flask API |
| 2 | Databases | [AmazonReviewsAnalysis](https://github.com/iqueipopg/AmazonReviewsAnalysis) | Amazon reviews across MySQL, MongoDB and Neo4j |
| 3 | Natural Language Processing | [NerD](https://github.com/iqueipopg/NerD) | Multitask BiLSTM for named entities and sentiment in news |
| 4 | Reinforcement Learning | [DiSpAtCh](https://github.com/iqueipopg/DiSpAtCh) | Double DQN agent for warehouse navigation and delivery |

My final degree project, [SAM](https://github.com/iqueipopg/SAM-overview), automates systematic reviews and meta-analyses with nine LLM agents.

## What is not here

Assignment statements, course test suites, exams, datasets and trained weights belong to the course or to their original authors and are not redistributed. See [NOTICE.md](NOTICE.md).
