# Transformer from scratch

Natural Language Processing II, year 4 of the Mathematical Engineering and AI degree, ICAI (Fall 2025).

Encoder-decoder Transformer built from its parts: scaled dot-product attention heads, multi-head attention, feed-forward blocks, encoder and decoder layers with causal masking.

## What it does

- **`AttentionHead`** and **`MultiHeadAttention`**: scaled dot-product attention with optional masks.
- **`TransformerEncoderLayer` / `TransformerDecoderLayer`**: self-attention, cross-attention in the decoder, feed-forward block, residual connections and layer normalisation.
- **`TransformerEncoder` / `TransformerDecoder`** stacks and the full **`Transformer`**, with causal masking of the target sequence.

## Run

```bash
pip install -r requirements.txt
python -m src.transformer    # builds the model and runs a forward pass on random tokens
```

## Authorship

Built on a starter repository from the course staff. [CONTRIBUTION.md](CONTRIBUTION.md) lists which files I changed and links the commits, so my part can be read on its own.
