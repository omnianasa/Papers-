# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## 1. What is BERT?
BERT stands for **Bidirectional Encoder Representations from Transformers**.  
It is a language model that learns from large amounts of text by looking at both the words before and after each word in a sentence. This “bidirectional” learning helps it understand context more deeply than earlier models.  
Once trained, BERT can be adapted to many tasks—like question answering, text classification, and sentence-pair tasks—by adding just one output layer, without changing the main architecture.

## 2. How BERT is Pre-trained
BERT uses two main pre-training tasks on unlabeled text:
- **Masked Language Model (MLM)**: Randomly hide (mask) some words in the input and train the model to predict them using the surrounding words. This lets the model learn from both left and right context at once.
- **Next Sentence Prediction (NSP)**: Give the model two sentences and ask it to predict if the second sentence follows the first in the original text. This teaches the model to understand relationships between sentences.

## 3. BERT Architecture Explained
BERT is built on the **Transformer encoder** architecture introduced by Vaswani et al. (2017).  
Its key features are:

- **Layers (Transformer Blocks)**:  
  - `BERT-BASE`: 12 layers (Transformer blocks)  
  - `BERT-LARGE`: 24 layers  
- **Hidden Size**: The size of each layer’s output vector (768 for BASE, 1024 for LARGE).  
- **Attention Heads**: Multiple “attention” sub-layers in each block that focus on different relationships between words (12 heads in BASE, 16 in LARGE).  
- **Feed-forward Layers**: After attention, each block has a fully connected network for deeper processing.
  
- **Bidirectional Self-Attention**: Unlike GPT, where each word can only see previous words, BERT’s attention looks both left and right in every layer, giving a richer context.

- **Input Representation**:  
  Every input token is represented as the sum of three embeddings:  
  1. **Token Embedding** – the meaning of the word or subword (WordPiece vocabulary of 30,000 tokens).  
  2. **Segment Embedding** – indicates if a token belongs to sentence A or sentence B.  
  3. **Position Embedding** – shows where the token appears in the sequence.

- **Special Tokens**:  
  - `[CLS]`: Placed at the start of the sequence; used for classification outputs.  
  - `[SEP]`: Separator between sentences or end of a single sentence.

- **Unified for All Tasks**: The same architecture is used for pre-training and for fine-tuning on different tasks; only the output layer changes.

In summary, BERT is a stack of bidirectional Transformer encoder layers that convert input text into rich contextual representations, ready to be adapted to many NLP problems.

## 4. What Makes BERT Better?
Earlier models like ELMo and GPT are limited because they only look in one direction (left-to-right or right-to-left) or combine two one-way models.  
BERT is **deeply bidirectional**—it learns from both directions at every layer.  
This leads to better understanding for tasks that need full context, reduces the need for task-specific architectures, and achieves state-of-the-art results across many benchmarks.

## 5. Performance Highlights
BERT set new records on multiple NLP benchmarks when it was released:
- **GLUE score**: 80.5% (previous best: 72.8%)
- **MultiNLI accuracy**: 86.7%
- **SQuAD v1.1 F1 score**: 93.2
- **SQuAD v2.0 F1 score**: 83.1
These results showed strong improvements over previous models, even on small datasets.

## 6. How to Use This Paper
1. **Understand the objectives**: Learn why MLM and NSP help create powerful general-purpose language representations.
2. **Download pretrained BERT**: The authors provide models like `BERT-BASE` and `BERT-LARGE` on GitHub.
3. **Fine-tune for your task**: Add a simple output layer for classification, QA, or tagging, and fine-tune all parameters on your dataset.
4. **Replicate benchmarks**: Use GLUE, SQuAD, or your own dataset to test and compare results.
5. **Extend the approach**: Adapt MLM and NSP pre-training to your domain-specific data or build new tasks on top of BERT.

## 7. Resources
- Paper: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- GitHub: [https://github.com/google-research/bert](https://github.com/google-research/bert)
