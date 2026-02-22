import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from forward_pass import *  # Import all forward pass functions


def backpropagation(probabilities, vals, labels, weights, kernels):
    """
    Vectorized backpropagation for:
    Conv1 → ReLU → MaxPool1 → Conv2 → ReLU → MaxPool2 → Flatten → Dense → Softmax
    """
    batch_size = len(labels)
    

    # 1. Softmax gradient
    y = np.zeros_like(probabilities)
    y[np.arange(batch_size), labels] = 1
    dL_dsoft = probabilities - y  # (batch, n_classes)
    
    # 2. Dense layer gradients
    flattened = vals['flattened']  # (batch, flatten_dim)
    dW = flattened.T @ dL_dsoft
    db = np.sum(dL_dsoft, axis=0, keepdims=True)
    

    # 3. Flatten layer backward
    d_flat = dL_dsoft @ weights.T  # (batch, pooled2_h * pooled2_w)
    pooled2 = vals['p2']
    d_pool2 = d_flat.reshape(pooled2.shape)  # match MaxPool2 output shape
    
    # 4. MaxPool2 backward
    # MaxPool2 indices
    conv2 = vals['z2']
    pool_size = 2
    b, h, w = conv2.shape
    ph, pw = h // pool_size, w // pool_size
    windows2 = conv2.reshape(b, ph, pool_size, pw, pool_size)
    flat_windows2 = windows2.reshape(b, ph, pw, pool_size*pool_size)
    max_idx2 = np.argmax(flat_windows2, axis=-1)
    max_h2 = max_idx2 // pool_size
    max_w2 = max_idx2 % pool_size

    d_conv2 = np.zeros_like(conv2)
    batch_idx = np.repeat(np.arange(b), ph*pw)
    h_idx_flat = ((np.arange(ph)[:, None]*pool_size) + max_h2).ravel()
    w_idx_flat = ((np.arange(pw)[None, :]*pool_size) + max_w2).ravel()
    flat_idx = np.ravel_multi_index((batch_idx, h_idx_flat, w_idx_flat), conv2.shape)
    d_conv2.ravel()[flat_idx] = d_pool2.ravel()

    # 5. ReLU2 backward
    d_conv2_pre = d_conv2 * (conv2 > 0)
    

    # 6. Kernel2 gradient (conv2 weights)
    # a1 = input to conv2 (after pool1)
    a1 = vals['p1']  # pooled1

    k2 = kernels[1]
    k_size2 = k2.shape[0]

    # Sliding window view of input to conv2
    windows2 = sliding_window_view(a1, (k_size2, k_size2), axis=(1,2))
    b, h, w, kh, kw = windows2.shape

    # d_conv2_pre already has shape (b, h, w)
    d_conv2_flat = d_conv2_pre.reshape(-1)  # flatten batch + spatial
    windows2_reshaped = windows2.reshape(b*h*w, kh*kw)

    dK2 = (windows2_reshaped.T @ d_conv2_flat).reshape(kh, kw) / batch_size
    
 
    # 7. Transposed convolution for pool1
    # Backprop through conv2 to pool1
    pad = k_size2 - 1
    padded_grad = np.pad(d_conv2_pre, ((0,0),(pad,pad),(pad,pad)), mode='constant')
    windows_pad = sliding_window_view(padded_grad, (k_size2, k_size2), axis=(1,2))
    d_pool1 = np.einsum('bhwij,ij->bhw', windows_pad, k2)
    
 
    # 8. MaxPool1 backward
    conv1 = vals['z1']
    b, h1, w1 = conv1.shape
    ph1, pw1 = h1 // pool_size, w1 // pool_size
    reshaped1 = conv1.reshape(b, ph1, pool_size, pw1, pool_size)
    flat1 = reshaped1.reshape(b, ph1, pw1, pool_size*pool_size)
    max_idx1 = np.argmax(flat1, axis=-1)
    max_h1 = max_idx1 // pool_size
    max_w1 = max_idx1 % pool_size

    d_conv1 = np.zeros_like(conv1)
    batch_idx1 = np.repeat(np.arange(b), ph1*pw1)
    h_idx1_flat = ((np.arange(ph1)[:, None]*pool_size) + max_h1).ravel()
    w_idx1_flat = ((np.arange(pw1)[None, :]*pool_size) + max_w1).ravel()
    flat_idx1 = np.ravel_multi_index((batch_idx1, h_idx1_flat, w_idx1_flat), conv1.shape)
    d_conv1.ravel()[flat_idx1] = d_pool1.ravel()

 
    # 9. ReLU1 backward
    b, h1, w1 = conv1.shape
    pool_size = 2
    ph1, pw1 = h1 // pool_size, w1 // pool_size

    # Initialize d_conv1_pre with zeros of correct shape
    d_conv1_pre = np.zeros_like(conv1)  # (batch, h1, w1)

    # Use MaxPool1 indices to place gradients
    batch_idx = np.repeat(np.arange(b), ph1 * pw1)
    h_idx_flat = ((np.arange(ph1)[:, None] * pool_size) + max_h1).ravel()
    w_idx_flat = ((np.arange(pw1)[None, :] * pool_size) + max_w1).ravel()
    flat_idx = np.ravel_multi_index((batch_idx, h_idx_flat, w_idx_flat), conv1.shape)

    # Place the gradients from d_pool1 (shape: batch, ph1, pw1) into d_conv1_pre
    d_conv1_pre.ravel()[flat_idx] = d_pool1.ravel()

    # Apply ReLU derivative
    d_conv1_pre = d_conv1_pre * (conv1 > 0)
    

    # 10. Kernel1 gradient (conv1 weights)
    input_to_conv1 = vals['input_to_conv1']  # Make sure this is the original input to conv1
    k1 = kernels[0]
    k_size1 = k1.shape[0]

    # Sliding window view of input images
    windows1 = sliding_window_view(input_to_conv1, (k_size1, k_size1), axis=(1, 2))
    b, h_win, w_win, kh, kw = windows1.shape  # h_win, w_win should equal h1, w1

    # Verify dimensions match
    assert h_win == h1 and w_win == w1, f"Window dimensions ({h_win},{w_win}) don't match conv1 dimensions ({h1},{w1})"

    windows1_reshaped = windows1.reshape(b * h1 * w1, kh * kw)
    d_conv1_flat = d_conv1_pre.reshape(b * h1 * w1)

    dK1 = (windows1_reshaped.T @ d_conv1_flat).reshape(kh, kw) / batch_size

    # Return all gradients
    return {
        'dW': dW,
        'db': db,
        'dkernel1': dK1,
        'dkernel2': dK2
    }