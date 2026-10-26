# Skip-gram with negative sampling

Natural Language Processing, year 3 of the Mathematical Engineering and AI degree, ICAI (Spring 2025).

Word2vec skip-gram trained with negative sampling on text8: subsampling of frequent words, context windows, the negative-sampling loss and a cosine-similarity check of the learned embeddings.

## What it does

- Tokenisation, vocabulary lookup tables and **subsampling** of frequent words with the rule of Mikolov et al. (2013), `1 - sqrt(t / f(w))`.
- Random-size **context windows** and a batch generator of (centre, context) pairs.
- **`SkipGramNeg`**: separate input and output embeddings, and noise words drawn from the unigram distribution raised to 3/4.
- **`NegativeSamplingLoss`**: log-sigmoid of the true pair plus the log-sigmoid of the negated noise pairs.
- A validation step that prints the nearest neighbours of sample words by cosine similarity during training.

## Run

```bash
pip install -r requirements.txt
# download text8 (http://mattmahoney.net/dc/text8.zip) and unzip it to data/text8
python -m src.main
```

## Authorship

Built on a starter repository from the course staff. [CONTRIBUTION.md](CONTRIBUTION.md) lists which files I changed and links the commits, so my part can be read on its own.
