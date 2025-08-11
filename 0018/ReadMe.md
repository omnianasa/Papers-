
# Attention Is All You Need (Transformer)

## Overview 
Today we explain the 2017 paper *Attention Is All You Need* in clear, everyday English. The paper introduces the **Transformer**, a neural network architecture for processing sequences (like sentences) that uses only attention mechanisms and removes recurrence (RNNs/LSTMs) and convolutions. The Transformer is faster to train, more parallelizable, and achieves state-of-the-art results on machine translation tasks.

## High-level idea
Traditional sequence models (RNNs) process tokens one-by-one in order. This makes training slow because you can't compute all token representations in parallel. The Transformer replaces that sequential processing with **self-attention**, which lets each position in the sequence look at all other positions at once and decide how much to focus on them. That means the model can learn long-range relationships directly, and training can be parallelized across the sequence.

## Main components 
- **Input token embeddings**: map tokens (words/subwords) to vectors.
- **Positional encodings**: add information about token order because the model has no recurrence.
- **Encoder stack**: several identical layers; each layer has multi-head self-attention + feed-forward network.
- **Decoder stack**: similar, but each layer has an extra encoder-decoder attention sub-layer to look at encoder outputs.
- **Output softmax**: convert decoder outputs to probabilities over vocabulary.

## Why attention? 
Self-attention computes a weighted average of all token representations, where weights depend on how relevant tokens are to each other. This gives a direct path between any two tokens (constant number of steps), making it easier to learn dependencies even across long distances. Multi-head attention lets the model look at relationships from different "perspectives" at once.

## Detailed pipeline — step by step 
1. **Data & tokenization**: Start with sentence pairs (source and target languages). Tokenize texts into subword units (e.g., byte-pair encoding or word-piece). This reduces vocabulary size and handles rare words.

2. **Create input embeddings**: Each token is mapped to a `d_model`-dimensional vector using learned embeddings. The same is done for the target side (decoder input) during training (teacher forcing).

3. **Add positional encoding**: Because the network doesn’t process tokens sequentially, we add a vector representing each token’s position to its embedding. The original paper uses sinusoidal functions of different frequencies, but learned positional embeddings also work.

4. **Encoder stack**: The encoder has N identical layers (N=6 in the base model). Each layer contains:
   - **Multi-head self-attention**: every position attends to all positions in the input sequence. This produces new contextualized vectors for each position.
   - **Feed-forward network**: a small 2-layer MLP applied independently at each position.
   Residual connections and layer normalization wrap each sub-layer (i.e., output = LayerNorm(x + Sublayer(x))). Dropout is applied for regularization.

5. **Decoder stack**: The decoder also has N layers. Each decoder layer contains:
   - **Masked multi-head self-attention**: the decoder attends only to previous positions (mask future tokens) to preserve auto-regressive generation.
   - **Encoder-decoder multi-head attention**: the decoder queries the encoder outputs to incorporate information from the source sentence.
   - **Feed-forward network**. Again, residual connections + layer normalization are used.

6. **Linear + softmax**: Decoder outputs are projected to vocabulary size and softmaxed to produce probabilities for the next token.

7. **Training loop**: Model is trained to predict next tokens on the target side using teacher forcing. The authors use Adam optimizer with a specific learning rate schedule (warm-up then inverse sqrt decay), label smoothing, and dropout.

8. **Inference**: At test time, generation is auto-regressive: the model produces one token at a time, feeding predicted tokens back into the decoder input. Beam search is commonly used for better results.

## Key formulas 
- **Scaled Dot-Product Attention**: `Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V`.
  - Q (queries), K (keys), and V (values) are matrices. The division by `sqrt(d_k)` scales dot-products to keep gradients stable.

- **Multi-Head Attention**: Project Q, K, V h times into smaller dimensions, apply attention in parallel, then concatenate results and project back.

- **Feed-Forward (position-wise)**: `FFN(x) = max(0, x W1 + b1) W2 + b2`, applied independently at each sequence position.

## Pseudocode 
```
# single attention head (matrix-friendly)
def scaled_dot_product_attention(Q, K, V):
    scores = Q @ K.T / sqrt(d_k)
    weights = softmax(scores)
    return weights @ V

# multi-head
def multi_head_attention(X):
    heads = []
    for i in range(h):
        Q = X @ Wq[i]
        K = X @ Wk[i]
        V = X @ Wv[i]
        heads.append(scaled_dot_product_attention(Q, K, V))
    concat = concat_heads(heads)
    return concat @ Wo
```

## Typical hyperparameters 
- `d_model` (embedding size): 512 (base) or 1024 (big)
- `d_ff` (hidden size in FFN): 2048 (base) or 4096 (big)
- `h` (number of attention heads): 8 (base) or 16 (big)
- Number of layers `N`: 6 (base)
- Dropout: 0.1 (base)
- Optimizer: Adam with warmup steps = 4000 and learning rate schedule `lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})`

## Training tips and tricks 
- **Batching**: Group sentences by similar length to avoid padding waste. The paper batches by token counts (e.g., 25k source tokens + 25k target tokens per batch).
- **Checkpoint averaging**: Average the last several checkpoints before evaluation to get more stable performance.
- **Label smoothing**: Use small label smoothing (0.1) to improve BLEU.
- **Masking**: Carefully apply masks (padding mask for encoder, look-ahead mask for decoder) to avoid letting the model attend to padded positions or future tokens.

## Inference details
- Use greedy decoding for fast outputs or beam search (beam size 4—8) for higher-quality translation.
- The model sets a maximum output length (e.g., input length + 50) to prevent runaway generation.

## Intuition visualized (what attention learns)
Attention heads often learn linguistically meaningful behaviors: some heads focus on local context (nearby words), others pick up long-distance relations (subject–verb), and some seem to capture syntactic roles (determiners, noun phrases). This makes the model more interpretable than some RNNs.

## Where to start implementing (practical steps)
1. Implement scaled dot-product attention and test it on random tensors.
2. Implement multi-head attention and verify shapes.
3. Build a single encoder layer (multi-head + FFN + residual + layernorm).
4. Stack N encoder layers and test with dummy inputs.
5. Build the decoder with masking and encoder-decoder attention.
6. Add embeddings + positional encodings and a training loop.
7. Train on a small dataset (toy translation) to verify the pipeline before scaling up.

## Common variations and extensions
- Replace sinusoidal positional encodings with **learned** embeddings.
- Use **relative positional encodings** for better handling of variable lengths.
- Use **local / restricted attention** when handling very long sequences.
- Larger models, more heads, and mixture-of-experts layers have been used to scale up performance.

---
To get and read the paper check [this](https://arxiv.org/pdf/1706.03762)