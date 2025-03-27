import numpy as np
from mnist_data import main

'''
Colin Johnson
Brinton Higgins
'''

(images_train, labels_train), (images_test, labels_test) = main()
    

# Each photo is 28x28

def generate_kernel(kernel_size):
    kernel = np.random.randn(kernel_size, kernel_size) * 0.1
    return kernel

def convolution(matrix, kernel):
    '''
    Performs convolution and activation all in one function
    matrix: input_matrix
    kernel: the kernel matrix used in convolution
    '''
    # Step 1: Convolutional of Matrix
    
    batch_size, matrix_height, matrix_width = matrix.shape
    kernel_height, kernel_width = kernel.shape

    output_height = matrix_height - kernel_height + 1
    output_width = matrix_width - kernel_width + 1

    #print(output_height)
    conv_layer1 = np.zeros((batch_size, output_height, output_width))
    for x in range(batch_size):
        for i in range(output_height):
        
            for j in range(output_width):
       
                submatrix = matrix[x, i:i+kernel_height, j:j+kernel_height]

                conv_layer1[x, i,j] = np.sum(submatrix * kernel)

    ReLU = np.maximum(0, conv_layer1)

    return conv_layer1, ReLU                



# Step 2 Max pooling

def max_pooling(matrix, pool_size):
    '''
    Performs max pooling on specific convolution layers
    matrix: input matrix
    pool_size: the size in which we pool by
    '''
    batch_size, m, n = matrix.shape

    output_height = m // pool_size
    output_width = n // pool_size

    pooled_matrix = np.zeros((batch_size, output_height, output_width))

    for x in range(batch_size):
        for i in range(output_height):
            for j in range(output_width):

                i_start, i_end = i*pool_size, (i+1)*pool_size
                j_start, j_end = j*pool_size, (j+1)*pool_size

                pooled_matrix[x, i, j] = np.max(matrix[x, i_start:i_end, j_start:j_end])

    return pooled_matrix

# Step 3 Dense layer
def initialize_parameters(input_size, output_size):
    """Initialize weights and biases for dense layer"""
    '''
    Performs initial calculations for dense layer, getting the weights and the bias martices
    input_size: size of the flattened matrix 
    output_size: num of classes we are working with
    '''
    scale = np.sqrt(2.0 / (input_size + output_size)) # Xavier initialization
    weights = np.random.randn(input_size, output_size) * scale  # Small random values
    biases = np.zeros((1, output_size))
  
    return weights, biases

def dense_layer(input_vector, weights, biases):
    return np.dot(input_vector, weights) + biases

def softmax(x):
    '''
    Implement softmax different as with the base formula if theta tranpose * x is too large then overlow
    This is similar implementation where e^(x-max(x)) ensures there is no overflow
    '''
    x = x - np.max(x, axis=1, keepdims=True)  # Prevent overflow
    exp_logits = np.exp(x)
    prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return prob

def forward_pass(image, kernels, weights, biases):
    '''
    
    '''
    # Conv + ReLU

    conv1, ReLU1 = convolution(image, kernels[0])
    pooled_1 = max_pooling(conv1, 2) # 12x12

    conv2, ReLU2 = convolution(pooled_1, kernels[1])
    pooled_2 = max_pooling(conv2, pool_size=2)
    
    # Flatten
    flattened = pooled_2.reshape(-1, 16)
    
    # Dense layer
    logits = dense_layer(flattened, weights, biases)
    
    # Softmax
    prob = softmax(logits)
    
    vals = {
        "z1": conv1, "a1": ReLU1, "p1": pooled_1,
        "z2": conv2, "a2": ReLU2, "p2": pooled_2,
        'flattened': flattened
    }

    return prob, vals

def cross_entropy_loss(probabilities, labels):
    '''
    Find the loss of the model
    probabilities: the array of output probabilities after forward pass
    labels: the array of labels of the correct number
    '''
    # Discover the number of labels in the dataset
    n = len(labels)
    # this variable looks at the numpy array of probabilities 
    # and finds the position of the correct number and takes the probability of it
    probabilities_of_right = np.zeros_like(probabilities)
    probabilities_of_right[np.arange(n), labels] = 1
    # add 1e-10 to avoid log(0) error
    return -np.sum(np.log(probabilities_of_right + 1e-10)) / n

def backpropagation(probabilities, vals, labels, weights, kernel):
    """
    prob: Final softmax probabilities (shape: (batch, 10))
    cache: Dictionary containing intermediate values
    y_true: True labels (shape: (batch,))
    """
    # conv1 <- ReLU1 <- pool1 <- conv2 <- ReLU2 <- pool2 <- Flatten <- Dense <- Softmax
    n = len(labels)

    conv1 = vals['z1']
    conv2 = vals['z2']
    
    # start backwards so find the Loss at softmax
    y = np.zeros_like(probabilities)
    y[np.arange(n), labels] = 1
    a1 = vals['a1']

    p2 = vals['p2']

    # full proof in github
    # softmax
    dL_dsoftinputs = probabilities - y

    flattened_input = vals['flattened']
    # Dense layer
    dL_dW = flattened_input.T @ dL_dsoftinputs
    print(dL_dW)
    dL_db = np.sum(dL_dsoftinputs, axis=0, keepdims=True)

    #Flatten
    dL_dflattened = dL_dsoftinputs @ weights.T
    '''
    # Pool 2, Conv2, and ReLU
    pooled_2 = vals['pooled_2']
    dL_dpool2 = dL_dflattened.reshape(pooled_2.shape)

    # According to google
    # Max-pooling is non-differentiable, 
    # but we can approximate its gradient by routing gradients only to the max values in the original pooling windows.

    dL_dconv2 = np.zeros_like(conv2)

    pool_size = 2  
    batch_size, h, w = conv2.shape

    # Here we will essentially "undo" the matrix and only take the gradient of the parts with a max value
    for b in range(batch_size):
        for i in range(0, h, pool_size):
            for j in range(0, w, pool_size):
                window = conv2[b, i:i+pool_size, j:j+pool_size]
                max_idx = np.unravel_index(np.argmax(window), window.shape)
                dL_dconv2[b, i + max_idx[0], j + max_idx[1]] = dL_dpool2[b, i//pool_size, j//pool_size]

    dL_dconv2_pre_relu = dL_dconv2 * (conv2 > 0)
    
    dL_dkernel2 = np.zeros_like(kernel[1])  # e.g., (5, 5)
    conv2_input = a1
    kernel_size = 5

    for b in range(batch_size):
        for i in range(h - kernel_size + 1):
            for j in range(w - kernel_size + 1):
                window = conv2_input[b, i:i+kernel_size, j:j+kernel_size]
                dL_dkernel2 += window * dL_dconv2_pre_relu[b, i, j]
    
    dL_dkernel2 /= batch_size

    
    # 8. Gradient through pool1 (conv2 connects to pool1 through weights)
    # Need to compute gradient w.r.t. pool1 output
    # This is more complex because conv2 was applied to pool1's output
    # We need to perform a full convolution transpose operation
    
    # Initialize gradient for pool1 output
    dL_dpool1 = np.zeros_like(p1)
    kernel2 = kernels[1]
    k = kernel2.shape[0]
    
    # This is the convolution transpose (sometimes called "full convolution")
    for b in range(batch_size):
        for i in range(dL_dpool1.shape[1]):
            for j in range(dL_dpool1.shape[2]):
                # The receptive field of conv2 that influenced pool1[i,j]
                i_start = max(0, i - k + 1)
                j_start = max(0, j - k + 1)
                i_end = min(i + 1, dL_dconv2_pre_relu.shape[1])
                j_end = min(j + 1, dL_dconv2_pre_relu.shape[2])
                
                for di in range(i_start, i_end):
                    for dj in range(j_start, j_end):
                        dL_dpool1[b, i, j] += np.sum(
                            kernel2[i-di+k-1, j-dj+k-1] * 
                            dL_dconv2_pre_relu[b, di, dj]
                        )
    
    # 9. Gradient through max-pool1 (route to max positions)
    dL_dconv1 = np.zeros_like(z1)
    batch_size, h, w = z1.shape
    
    for b in range(batch_size):
        for i in range(0, h, pool_size):
            for j in range(0, w, pool_size):
                window = z1[b, i:i+pool_size, j:j+pool_size]
                max_idx = np.unravel_index(np.argmax(window), window.shape)
                dL_dconv1[b, i + max_idx[0], j + max_idx[1]] = dL_dpool1[b, i//pool_size, j//pool_size]
    
    # 10. Gradient through ReLU1
    dL_dconv1_pre_relu = dL_dconv1 * (z1 > 0)
    
    # 11. Gradient for kernel1
    dL_dkernel1 = np.zeros_like(kernels[0])
    conv1_input = vals['input']  # Original input to the network
    
    for b in range(batch_size):
        for i in range(conv1_input.shape[1] - kernel_size + 1):
            for j in range(conv1_input.shape[2] - kernel_size + 1):
                window = conv1_input[b, i:i+kernel_size, j:j+kernel_size]
                dL_dkernel1 += window * dL_dconv1_pre_relu[b, i, j]
    
    dL_dkernel1 /= batch_size
    '''
    
def main():
    
    # Image sizes are 28x28x1 after performing images_train.shape()
    n_images = 1000

    subset_images = images_train[:n_images]  # Shape: (5, 28, 28) if already reshaped, or (5, 784)
    subset_labels = labels_train[:n_images]
    input_matrix =  subset_images.reshape(-1, 28, 28) / 255.0
    print(input_matrix.shape)

    # Layer definitions
    kernel1 = generate_kernel(5)  # 5x5
    kernel2 = generate_kernel(5)  # 2x2
    kernels = [kernel1, kernel2]

    # Initialize parameters
    num_classes = 10
    flattened_size = 16

    weights, biases = initialize_parameters(flattened_size, num_classes)

    prob, vals = forward_pass(input_matrix, kernels, weights, biases)
    
    backpropagation(prob, vals, subset_labels, weights, kernels)

main()