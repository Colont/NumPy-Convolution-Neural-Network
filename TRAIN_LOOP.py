from sklearn.datasets import fetch_openml
import numpy as np
from forward_pass import *
from backpropagation import *


if __name__ == '__main__':

    # Load data
    
    mnist = fetch_openml('mnist_784', version=1)
    images = mnist.data.values.astype(np.float32).reshape(-1, 28, 28)
    labels = mnist.target.astype(int)
    
    images /= 255.0

    # Split into train/test (60k train, 10k test)
    train_images = images[:60000]
    train_labels = labels[:60000]
    test_images = images[60000:]
    test_labels = labels[60000:]

    k1 = generate_kernel(5)
    k2 = generate_kernel(5)
    kernels = [k1, k2]

    flatten_size = ((28 - 5 + 1)//2 - 5 + 1)//2
    flatten_size = flatten_size * flatten_size

    weights, biases = initialize_parameters(flatten_size, 10)

    learning_rate = 0.01
    batch_size = 32
    epochs = 5

    for epoch in range(epochs):
        # Shuffle the dataset
        perm = np.random.permutation(len(train_images))
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        
        # Track accuracy for this epoch
        correct_predictions = 0
        total_predictions = 0
        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(train_images), batch_size):
            # Get a batch of images and labels
            batch_imgs = train_images[i:i+batch_size]
            batch_lbls = train_labels[i:i+batch_size]

            # Forward pass
            probs, vals = forward_pass(batch_imgs, kernels, weights, biases)

            # Get predictions (class with highest probability)
            predictions = np.argmax(probs, axis=1)
            
            # Calculate accuracy for this batch
            batch_correct = np.sum(predictions == batch_lbls)
            correct_predictions += batch_correct
            total_predictions += len(batch_lbls)
            
            # Compute loss (optional for monitoring)
            loss = cross_entropy_loss(probs, batch_lbls)
            epoch_loss += loss
            num_batches += 1

            # Backpropagation
            grads = backpropagation(probs, vals, batch_lbls, weights, kernels)

            # Update parameters
            weights -= learning_rate * grads['dW']
            biases -= learning_rate * grads['db']
            kernels[0] -= learning_rate * grads['dkernel1']
            kernels[1] -= learning_rate * grads['dkernel2']

        # Calculate epoch metrics
        epoch_accuracy = correct_predictions / total_predictions * 100
        avg_loss = epoch_loss / num_batches
        
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")

    # Optional: Calculate test accuracy after training
    test_correct = 0
    test_total = 0
    
    for i in range(0, len(test_images), batch_size):
        batch_imgs = test_images[i:i+batch_size]
        batch_lbls = test_labels[i:i+batch_size]
        
        probs, _ = forward_pass(batch_imgs, kernels, weights, biases)
        predictions = np.argmax(probs, axis=1)
        
        test_correct += np.sum(predictions == batch_lbls)
        test_total += len(batch_lbls)
    
    test_accuracy = test_correct / test_total * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")