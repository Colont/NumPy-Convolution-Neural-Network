# Image Classification CNN from Scratch

**A convolutional neural network built using only NumPy, with no deep learning libraries.** Every layer, including the forward pass and backpropagation, is implemented by hand to show how CNNs work under the hood.

---

## Highlights

- **Built from scratch:** Implemented convolution, ReLU, max pooling, and softmax using vectorized NumPy operations.
- **Custom architecture:** A 2-layer convolutional network with 5x5 kernels.
- **Hand-derived backpropagation:** Derived and implemented gradients for every layer using the chain rule and Jacobians.
- **3x faster training:** Vectorized the backpropagation step, speeding up training 3x.

## How It Works

1. **Convolution:** 5x5 kernels slide across the image to detect features like edges and shapes.
2. **ReLU:** Sets negative values to zero, letting the network learn nonlinear patterns.
3. **Max pooling:** Shrinks each feature map by keeping the strongest signal in each region.
4. **Softmax:** Converts the final outputs into class probabilities.
5. **Backpropagation:** Gradients flow backward through every layer using the chain rule, and the weights are updated to reduce the loss.

## Tech Stack

Python, NumPy
