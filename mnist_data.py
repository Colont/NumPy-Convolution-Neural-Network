from tensorflow.keras.datasets import mnist



def main():
    (images_train, labels_train), (images_test, labels_test) = mnist.load_data()
    
    return (images_train, labels_train), (images_test, labels_test)


if __name__ == '__main__':
    main()