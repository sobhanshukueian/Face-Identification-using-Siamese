# Face Identification using PyTorch on AT&T Face Identification Dataset
<div align="center">
 <img src="https://github.com/sobhanshukueian/Face-Identification-using-Siamese/assets/47561760/fadc4381-e890-4908-ba19-d2074aa0f177" alt="your-gif" width="400" height="300" />
</p>
  <strong>Face Identification Project using PyTorch on the AT&T Face Identification Dataset</strong>
</div>

## Introduction

Welcome to the Face Identification project! This repository contains an implementation of a face identification system using PyTorch on the AT&T Face Identification Dataset. The goal of this project is to build a deep-learning model that can accurately identify individuals based on their facial features.

# AT&T Face Identification Dataset

There are ten different images of each of 40 distinct subjects. For some subjects, the images were taken at different times, varying the lighting, facial expressions (open/closed eyes, smiling / not smiling), and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement). A preview image of the Database of Faces is available.

kaggle URL: https://www.kaggle.com/datasets/kasikrit/att-database-of-faces?resource=download 

## Some Configs for downloading from Kaggle : 

1. Go to your account, Scroll to the API section, and Click Expire API Token to remove previous tokens
2. Click on Create New API Token - It will download the kaggle.json file on your machine.
3. Upload file.

## Pairs generation

For training Siamese networks we need two kinds of pairs to feed the network:

*   **Positive pairs:** Similar pairs or a person's face image(label = 0).
*   **Negative pairs**  Dissimilar pairs or different person face images(label = 1).

I should have balanced positive and negative pairs, so I implemented the ```create_data()``` function that generates pairs. In this function, I choose an image randomly, then select randomly n(size of classes or persons) positive pairs, and then from each class, I choose an image as a negative pair, so the amount of positive and negative pairs are balanced and in each batch, we have both positive and negative pairs in balanced amount. 

![image](https://user-images.githubusercontent.com/47561760/193102226-2af72580-885f-42a4-ae4a-457f2f5e3388.png)


# Model

The model consists of convolutional layers, Leaky Relu as an activation function, BatchNormalization, and dropout for regularization.
BacthNorm after each conv block helps to normalize representations and prevent the model from collapsing and helps train faster. The model output is an embedding with a dynamic size that you can change according to the size of classes.

# Contrastive Loss
Contrastive loss is evaluating how good a job the Siamese network is distinguishing between the image pairs. The difference is subtle but incredibly important.
To break this equation down:
The Y value is our label. It will be 0 if the image pairs are of the same class, and it will be 1 if the image pairs are of a different class. The D_{w} variable is the Euclidean distance between the outputs of the sister network embeddings. The max function takes the largest value of 0 and the margin, m, minus the distance.

Actually, it tries to distance be zero if pairs are similar and be margin if dissimilar.

![image](https://user-images.githubusercontent.com/47561760/193100817-28d1ef81-5b87-4dda-919e-c7399bf8338c.png)

# Trainer Class
This class Does the main part of the code which is the training model, plots the training process, and saves the model each n epochs.

I Defined `Contrastive` Loss and `Adam` Optimizer with learning rate of 0.001 and 0.999 momentum.

It does each training step in `train_step` function and validation step in `val_step` function and the whole training process in 
`train` function.
 
## Some Configurations
 
*   You can set epoch size: `EPOCHS` and batch size: `BATCH_SIZE`.
*   Set `device` that you want to train the model on it: `device`(default runs on cuda if it's available)
*   You can set one of three `verboses` that prints info you want => 0 == nothing || 1 == model architecture || 2 == print optimizer || 3 == model parameters size.
*   Each time you train model weights and plot(if `save_plots` == True) will be saved in `save_dir`.
*   You can find a `configs` file in `save_dir` that contains some information about run.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code for any purpose.

---

We hope this project encourages you to explore face identification techniques using deep learning. If you have any questions or need further assistance, please don't hesitate to reach out.

Happy face identification and coding!

