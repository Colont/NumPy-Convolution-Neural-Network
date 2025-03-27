import numpy as np
from forward_pass import *  # Import all forward pass functions

def backpropagation(probabilities, vals, labels, weights, kernels):
    """
    Perform backpropagation through the network
    """
    n = len(labels)
    conv1 = vals['z1']
    conv2 = vals['z2']
    
    # Softmax gradient
    y = np.zeros_like(probabilities)
    y[np.arange(n), labels] = 1
    dL_dsoftinputs = probabilities - y

    # Dense layer gradients
    flattened_input = vals['flattened']
    dL_dW = flattened_input.T @ dL_dsoftinputs
    dL_db = np.sum(dL_dsoftinputs, axis=0, keepdims=True)

    # Flatten gradient
    dL_dflattened = dL_dsoftinputs @ weights.T

    # Pool2 gradient (approximate)
    pooled_2 = vals['p2']
    dL_dpool2 = dL_dflattened.reshape(pooled_2.shape)

    # Conv2 gradient
    dL_dconv2 = np.zeros_like(conv2)
    pool_size = 2  
    batch_size, h, w = conv2.shape

    for b in range(batch_size):
        for i in range(0, h, pool_size):
            for j in range(0, w, pool_size):
                window = conv2[b, i:i+pool_size, j:j+pool_size]
                max_idx = np.unravel_index(np.argmax(window), window.shape)
                dL_dconv2[b, i + max_idx[0], j + max_idx[1]] = dL_dpool2[b, i//pool_size, j//pool_size]

    dL_dconv2_pre_relu = dL_dconv2 * (conv2 > 0)
    
    # Kernel2 gradient
    dL_dkernel2 = np.zeros_like(kernels[1])
    conv2_input = vals['a1']
    kernel_size = 5

    for b in range(batch_size):
        for i in range(h - kernel_size + 1):
            for j in range(w - kernel_size + 1):
                window = conv2_input[b, i:i+kernel_size, j:j+kernel_size]
                dL_dkernel2 += window * dL_dconv2_pre_relu[b, i, j]
    
    dL_dkernel2 /= batch_size

    ''' 
    # 8. Gradient through pool1 (conv2 connects to pool1 through weights)
    # Need to compute gradient w.r.t. pool1 output
    # This is more complex because conv2 was applied to pool1's output
    # We need to perform a full convolution transpose operation
    
    # Initialize gradient for pool1 output
    
    # 9. Gradient through max-pool1 (route to max positions)
   
    
    # 10. Gradient through ReLU1

    
    # 11. Gradient for kernel1
   
    '''
    return None