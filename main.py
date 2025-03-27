import numpy as np
from forward_pass import *
from backpropagation import *
from mnist_data import main

def train_model():
    # Load data
    (images_train, labels_train), (images_test, labels_test) = main()
    
    # Preprocess
    n_images = 1000
    subset_images = images_train[:n_images]
    subset_labels = labels_train[:n_images]
    input_matrix = subset_images.reshape(-1, 28, 28) / 255.0

    # Initialize model
    kernels = [generate_kernel(5), generate_kernel(5)]
    weights, biases = initialize_parameters(16, 10)

    # Forward pass
    prob, vals = forward_pass(input_matrix, kernels, weights, biases)
    
    # Backward pass
    #gradients = backpropagation(prob, vals, subset_labels, weights, kernels)
    

if __name__ == "__main__":
    train_model()