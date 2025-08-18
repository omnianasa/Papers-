# Extracting and composing robust features with denoising autoencoders

## Overview
A **Denoising Autoencoder (DAE)** is a neural network that learns to reconstruct clean inputs from corrupted (noisy) versions. It helps in learning **robust features** for deep learning tasks.

**Key Idea:**  
By training on corrupted inputs, the model learns meaningful patterns instead of just memorizing the data.

---

## Step-by-Step Data Flow with Math

### 1. Clean Input (`x`)
- Original input (e.g., image pixels):

$$
\mathbf{x} = [x_1, x_2, ..., x_d] \in [0,1]^d
$$

- \(d\) = input dimension (e.g., number of pixels).

---

### 2. Corrupt Input (`\tilde{x}`)
- Randomly mask or add noise to some values:

$$
\tilde{\mathbf{x}} \sim q_{\mathcal{D}}(\tilde{\mathbf{x}}|\mathbf{x})
$$`

- Example:  
If 

$$
\mathbf{x} = [0.2, 0.8, 0.5] \quad \Rightarrow \quad \tilde{\mathbf{x}} = [0.2, 0.0, 0.5]
$$

- \(q_{\mathcal{D}}\) = corruption distribution (e.g., randomly mask 30% of values).

---

### 3. Encode (`y`)
- Map corrupted input to hidden representation:

$$
\mathbf{y} = \sigma(\mathbf{W} \tilde{\mathbf{x}} + \mathbf{b})
$$

- \(\mathbf{W}\): weight matrix (\(d' \times d\))  
- \(\mathbf{b}\): bias vector  
- \(\sigma\): sigmoid activation (\(\sigma(z) = \frac{1}{1+e^{-z}}\))  
- \(d'\) = size of hidden layer

---

### 4. Decode (`z`)
- Reconstruct input from hidden representation:

$$
\mathbf{z} = \sigma(\mathbf{W'} \mathbf{y} + \mathbf{b'})
$$

- Often: \(\mathbf{W'} = \mathbf{W}^T\) (tied weights)  
- \(\mathbf{b'}\) = decoder bias  

- Goal: \(\mathbf{z} \approx \mathbf{x}\)

---

### 5. Loss Calculation
- **Binary Cross-Entropy Loss** (for binary data):

$$
L_H(\mathbf{x}, \mathbf{z}) = -\sum_{k=1}^d \left[ x_k \log z_k + (1-x_k)\log(1-z_k) \right]
$$

- **Mean Squared Error** (for continuous data):

$$
L(\mathbf{x}, \mathbf{z}) = \|\mathbf{x} - \mathbf{z}\|^2
$$

---

### 6. Training (Gradient Descent)
- Update weights to minimize reconstruction error:

$$
\theta \leftarrow \theta - \eta \nabla_\theta L(\mathbf{x}, \mathbf{z})
$$

- \(\eta\) = learning rate  
- \(\theta\) = model parameters (\(\mathbf{W}, \mathbf{b}, \mathbf{W'}, \mathbf{b'}\))

---

## Example (MNIST Digits)

| Step     | Description          | Math Example |
|----------|-------------------|--------------|
| Input    | Clean digit        | \(\mathbf{x} = [0.0, 0.8, 0.5, ..., 0.1]\) |
| Corrupt  | Randomly mask      | \(\tilde{\mathbf{x}} = [0.0, 0.0, 0.5, ..., 0.1]\) |
| Encode   | Hidden rep.        | \(\mathbf{y} = \sigma(\mathbf{W}\tilde{\mathbf{x}} + \mathbf{b})\) |
| Decode   | Reconstruct input  | \(\mathbf{z} = \sigma(\mathbf{W'} \mathbf{y} + \mathbf{b'})\) |
| Loss     | Compare z to x     | \(L = \|\mathbf{x} - \mathbf{z}\|^2\) |
| Update   | Adjust weights     | \(\mathbf{W} \leftarrow \mathbf{W} - \eta \nabla_\mathbf{W} L\) |

---

## Pseudocode

```python
def denoising_autoencoder(x, corruption_level=0.3):
    """
    Simple Denoising Autoencoder pseudocode.
    """
    # 1. Corrupt input
    x_corrupted = corrupt(x, corruption_level)  # Randomly mask values
    
    # 2. Encode
    y = sigmoid(np.dot(W, x_corrupted) + b)
    
    # 3. Decode
    z = sigmoid(np.dot(W_prime, y) + b_prime)
    
    # 4. Compute loss
    loss = binary_cross_entropy(x, z)
    
    # 5. Update weights (gradient descent)
    W -= learning_rate * gradient(loss, W)
    b -= learning_rate * gradient(loss, b)
    W_prime -= learning_rate * gradient(loss, W_prime)
    b_prime -= learning_rate * gradient(loss, b_prime)
    
    return z

    ```

Check this paper to read more [DOI:10.1145/1390156.1390294](https://dl.acm.org/doi/10.1145/1390156.1390294)
