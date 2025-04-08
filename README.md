# Vision Transformer (ViT)

## Credits & Acknowledgements
This project is a reimplementaion of [ViT-PyTorch] by [explainingai-code] (https://github.com/explainingai-code/VIT-Pytorch)
The code has been rewritten from scratch while maintaining the core concepts and functionalities of the original implementation.

## Features
- Build a Vision Transformer that can be mofified using a config file with a compatible format.
- The code showcased a Vision Transformer trained on a modified MNIST dataset with various background, in the size of 224x224.

## Description of Files:
- **extract_mnist.py** - Extracts MNIST data from CSV files.
- **custom_data.py** - Create a custom MNIST dataset with various background.
- **model.py** - Compatible with .yaml config files to create various Vision Transformer models.
- **engine.py** - Defines the train and test steps (for 1 epoch).
- **main.py** - Trains a Vision Transformer.
- **inference.py**
  1. Plot a confusion matrix
  2. Plot the attention maps
  3. Plot the similarity between positional embeddings.
