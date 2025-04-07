import numpy as np
from forward_pass import *  # Import all forward pass functions

def backpropagation(probabilities, vals, labels, weights, kernels):
    """
    Perform backpropagation through the network with mathematical explanations
    
    Network architecture:
    Conv1 → ReLU → MaxPool1 → Conv2 → ReLU → MaxPool2 → Flatten → Dense → Softmax
    """
    n = len(labels)
    conv1 = vals['z1']  # Output of first convolution (before ReLU)
    conv2 = vals['z2']  # Output of second convolution (before ReLU)
    
    # =====================================================================
    # 1. Softmax Gradient (Cross-Entropy Loss derivative)
    # ∂L/∂z = p_i - y_i (where p_i is predicted prob, y_i is true label)
    # =====================================================================
    y = np.zeros_like(probabilities)
    y[np.arange(n), labels] = 1  # One-hot encode true labels
    dL_dsoftinputs = probabilities - y  # Gradient at softmax layer

    # =====================================================================
    # 2. Dense Layer Gradients
    # ∂L/∂W = a_{L-1}^T · ∂L/∂z_L (where a is activation, z is pre-activation)
    # ∂L/∂b = sum(∂L/∂z_L) across batch
    # =====================================================================
    flattened_input = vals['flattened']  # Output from MaxPool2 (flattened)
    dL_dW = flattened_input.T @ dL_dsoftinputs  # Gradient for dense weights
    dL_db = np.sum(dL_dsoftinputs, axis=0, keepdims=True)  # Gradient for biases

    # =====================================================================
    # 3. Flatten Layer Gradient
    # ∂L/∂a_{L-1} = ∂L/∂z_L · W^T
    # =====================================================================
    dL_dflattened = dL_dsoftinputs @ weights.T
    
    # =====================================================================
    # 4. MaxPool2 Gradient (Backprop through max pooling)
    # Only the max position in each window receives gradient
    # ∂L/∂a_{pool} = upsample(∂L/∂a_{next}) with zeros except at max positions
    # =====================================================================
    pooled_2 = vals['p2']
    dL_dpool2 = dL_dflattened.reshape(pooled_2.shape)  # Reshape to match pool output
   
    # Initialize gradient for conv2 output (before pooling)
    dL_dconv2 = np.zeros_like(conv2)
    pool_size = 2  
    batch_size, h, w = conv2.shape

    for b in range(batch_size):
        for i in range(0, h, pool_size):
            for j in range(0, w, pool_size):
                window = conv2[b, i:i+pool_size, j:j+pool_size]
                max_idx = np.unravel_index(np.argmax(window), window.shape)
                # Only the max position gets gradient
                dL_dconv2[b, i + max_idx[0], j + max_idx[1]] = dL_dpool2[b, i//pool_size, j//pool_size]
    
    # =====================================================================
    # 5. ReLU Gradient for Conv2
    # ∂L/∂z = ∂L/∂a * (z > 0) where a = ReLU(z)
    # =====================================================================
    dL_dconv2_pre_relu = dL_dconv2 * (conv2 > 0)  # Gradient before ReLU

    # =====================================================================
    # 6. Kernel2 Gradient (Conv2 weights)
    # ∂L/∂K2 = a1 ∗ ∂L/∂z2 (valid cross-correlation)
    # Where ∗ is valid cross-correlation between input and gradient
    # =====================================================================
    dL_dkernel2 = np.zeros_like(kernels[1])
    conv2_input = vals['a1']  # Output from pool1 (after ReLU)
    kernel_size = kernels[1].shape[0]

    for b in range(batch_size):
        for i in range(h - kernel_size + 1):
            for j in range(w - kernel_size + 1):
                window = conv2_input[b, i:i+kernel_size, j:j+kernel_size]
                dL_dkernel2 += window * dL_dconv2_pre_relu[b, i, j]
    
    # Average gradient over batch
    dL_dkernel2 /= batch_size

    # =====================================================================
    # 7. Gradient for Pool1 Output (Backprop through Conv2)
    # ∂L/∂a1 = full_convolution(∂L/∂z2, rot180(K2))
    # This is a transposed convolution operation
    # =====================================================================
    dL_dpool1 = np.zeros_like(conv2_input)
    kernel2 = kernels[1]

    for b in range(batch_size):
        for i in range(h):
            for j in range(w):
                # Spread gradient through kernel (transposed conv)
                dL_dpool1[b, i:i+kernel_size, j:j+kernel_size] += \
                    kernel2 * dL_dconv2_pre_relu[b, i, j]
                
    # =====================================================================
    # 8. MaxPool1 Gradient (Same as MaxPool2)
    # =====================================================================
    dL_dconv1 = np.zeros_like(conv1)
    batch_size, h_p1, w_p1 = conv1.shape
    
    for b in range(batch_size):
        for i in range(0, h_p1, pool_size):
            for j in range(0, w_p1, pool_size):
                window = conv1[b, i:i+pool_size, j:j+pool_size]
                max_idx = np.unravel_index(np.argmax(window), window.shape)
                dL_dconv1[b, i + max_idx[0], j + max_idx[1]] = dL_dpool1[b, i//pool_size, j//pool_size]

    # =====================================================================
    # 9. ReLU Gradient for Conv1
    # =====================================================================
    dL_dconv1_pre_relu = dL_dconv1 * (conv1 > 0)

    # =====================================================================
    # 10. Kernel1 Gradient (Same as Kernel2)
    # ∂L/∂K1 = input ∗ ∂L/∂z1
    # =====================================================================
    dL_dkernel1 = np.zeros_like(kernels[0])
    input_images = vals['z1']
    kernel_size = kernels[0].shape[0]

    for b in range(batch_size):
        for i in range(h_p1 - kernel_size + 1):
            for j in range(w_p1 - kernel_size + 1):
                window = input_images[b, i:i+kernel_size, j:j+kernel_size]
                dL_dkernel1 += window * dL_dconv1_pre_relu[b, i, j]

    # Average gradient over batch

    dL_dkernel1 /= batch_size

    # Return all gradients

    gradients = {
        'dW': dL_dW,
        'db': dL_db,
        'dkernel1': dL_dkernel1,
        'dkernel2': dL_dkernel2
    }
    
    return gradients