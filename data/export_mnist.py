# Importing required libraries
from os import makedirs, path
from keras.datasets import mnist
from PIL import Image

# Loading the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Ensuring the shapes of the datasets
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Getting the current directory
dir = path.dirname(path.realpath(__file__))

# Function to initialize directory structure for saving images
def init_dir():
    for i in range(10):
        makedirs(f'{dir}/train/{i}', exist_ok=True)
        makedirs(f'{dir}/test/{i}', exist_ok=True)

# Function to save training images to their respective directories
def save_train():
    for i in range(60000):
        img = Image.fromarray(x_train[i])
        img.save(f'{dir}/train/{y_train[i]}/{i}.jpg')
        print(f'save train: {i}')

# Function to save testing images to their respective directories
def save_test():
    for i in range(10000):
        img = Image.fromarray(x_test[i])  # Corrected x_train to x_test
        img.save(f'{dir}/test/{y_test[i]}/{i}.jpg')  # Corrected y_train to y_test
        print(f'save test: {i}')

# Main function to initialize directories and save images
def main():
    init_dir()
    # save_train()
    save_test()

# Condition to call the main function when the script is executed
if __name__ == "__main__":
    main()