# Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling

## Introduction

Recurrent Neural Networks (RNNs) have proven to be highly effective for sequence modeling tasks such as machine translation, speech processing, and music generation. However traditional RNNs suffer from major limitations when it comes to learning long term dependencies due to vanishing or exploding gradients. This has led to the development of more advanced units designed to address these issues: the Long Short-Term Memory (LSTM) and the Gated Recurrent Unit (GRU).

Today we are providing a detailed explanation of the internal architectures of RNN, LSTM, and GRU units, based on the paper "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" by Chung et al., 2014. The goal is to understand the mathematical structure and logic behind each unit, not just to summarize.

---

## 1. Recurrent Neural Networks (RNN)

An RNN processes sequential input by maintaining a hidden state that is updated at each time step based on the current input and the previous hidden state.

Given an input sequence \( x = (x_1, x_2, ..., x_T) \), the RNN computes a hidden state \( h_t \) as:

\[
h_t = \phi(Wx_t + Uh_{t-1} + b)
\]

- \( W \) and \( U \) are learned weight matrices.
- \( \phi \) is a non-linear activation function, usually tanh or sigmoid.
- \( h_0 \) is typically initialized to a zero vector.

### Limitations

Traditional RNNs struggle with learning dependencies across long sequences due to vanishing gradients (where gradients become too small to update weights effectively if we used tanh or sigmoid) or exploding gradients (where gradients grow uncontrollably large if we used ReLU).

---

## 2. Long Short-Term Memory (LSTM)

LSTM was introduced to address the limitations of standard RNNs by introducing memory cells and gates that explicitly control what information to keep, write, or forget.

### Architecture

Each LSTM unit contains:

- A memory cell \( c_t \): carries long-term information.
- A hidden state \( h_t \): output at time \( t \).
- Three gates:
  - Input gate \( i_t \)
  - Forget gate \( f_t \)
  - Output gate \( o_t \)

### Computation Steps

Given input \( x_t \), previous hidden state \( h_{t-1} \), and previous memory \( c_{t-1} \):

1. **Forget gate**: controls what information to forget.
   \[
   f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
   \]

2. **Input gate**: controls what new information to add.
   \[
   i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
   \]
   \[
   \tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
   \]

3. **Update memory**:
   \[
   c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
   \]

4. **Output gate**: controls what information to output.
   \[
   o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
   \]
   \[
   h_t = o_t \odot \tanh(c_t)
   \]


LSTM allows the network to remember or forget information over long sequences and making it highly effective for tasks with long term dependencies.

---

## 3. Gated Recurrent Unit (GRU)


GRU is a simplified version of LSTM, proposed to reduce computational complexity while maintaining the ability to capture long-term dependencies. Unlike LSTM, GRU does not maintain a separate memory cell.

### Architecture

Each GRU unit includes:

- A hidden state \( h_t \)
- Two gates:
  - Reset gate \( r_t \)
  - Update gate \( z_t \)

### Computation Steps

Given input \( x_t \) and previous hidden state \( h_{t-1} \):

1. **Update gate**: decides how much of the past to keep.
   \[
   z_t = \sigma(W_z x_t + U_z h_{t-1})
   \]

2. **Reset gate**: decides how much past information to forget when calculating the candidate.
   \[
   r_t = \sigma(W_r x_t + U_r h_{t-1})
   \]

3. **Candidate hidden state**:
   \[
   \tilde{h}_t = \tanh(W x_t + U (r_t \odot h_{t-1}))
   \]

4. **Final hidden state**:
   \[
   h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
   \]

### Summary

GRU provides a more compact alternative to LSTM with fewer gates and no separate memory cell. It is faster to train and often performs comparably to LSTM on various tasks.

---

## A Comparison

| Feature                      | RNN (Tanh)        | LSTM                            | GRU                          |
|-----------------------------|-------------------|----------------------------------|------------------------------|
| Handles long-term memory    | No               | Yes(via memory cell)             | Yes (via gating mechanism)    |
| Gates used                  | None              | Input, Forget, Output            | Update, Reset                |
| Separate memory cell        | No               | Yes                              | No                         |
| Computational cost          | Low               | High                             | Moderate                     |
| Training time               | Fast              | Slow                             | Faster than LSTM             |
| Performance on long sequences| Poor              | Excellent                        | Very good                    |

--- 
[Paper on arxiv](https://arxiv.org/pdf/1412.3555)

