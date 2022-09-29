# Siamese Face Identification on AT&T
Face identification is the task of matching a given face image to one in an existing database of faces.So how can we solve this problem with siamese networks?

## Siamese Networks
Siamese is a kind of network that consists of two identical networks, these networks map inputs to a latent space that can compare them in that space. To train these networks, we use losses that measure the similarity between two vectors in space. The network learns to identify similar and dissimilar inputs according to the distribution of representations of the inputs.
So the problem is solved; we can use Siamese networks to identify different faces.

![image](https://user-images.githubusercontent.com/47561760/193100104-6515dc8f-f7b6-4e22-afd1-437456bd80b4.png)

# AT&T Face Identification Dataset

There are ten different images of each of 40 distinct subjects. For some subjects, the images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement). A preview image of the Database of Faces is available.

kaggle URL: https://www.kaggle.com/datasets/kasikrit/att-database-of-faces?resource=download 

## Some Configs for downloading from kaggle : 

1. Go to your account, Scroll to API section and Click Expire API Token to remove previous tokens
2. Click on Create New API Token - It will download kaggle.json file on your machine.
3. Upload file.

## Pairs generation

For trianing siamese networks we need two kind of pairs to feed network:

*   **Positive pairs:** Similar pairs or a person's face image(label = 0).
*   **Negative pairs**  Dissimilar pairs or different person face images(label = 1).

I should have balanced positive and negative pairs, so I implemented the ```create_data()``` function that generates pairs. In this function, I choose an image randomly, then select randomly n(size of classes or persons) positive pairs, and then from each class, I choose an image as a negative pair, so the amount of positive and negative pairs are balanced and in each batch we have both positive and negative pairs in balanced amount. 

![image](https://user-images.githubusercontent.com/47561760/193102226-2af72580-885f-42a4-ae4a-457f2f5e3388.png)


# Model

The model consists of convolutional layers, Leaky Relu as activation function, BatchNormalization, and dropout for regularization.
BacthNorm after each conv block helps to normalize representations and prevent the model from collapsing and helps train faster. The model output is an embedding with a dynamic size that you can change according to the size of classes.

# Contrastive Loss
Contrastive loss is evaluating how good a job the siamese network is distinguishing between the image pairs. The difference is subtle but incredibly important.
To break this equation down:
The Y value is our label. It will be 0 if the image pairs are of the same class, and it will be 1 if the image pairs are of a different class. The D_{w} variable is the Euclidean distance between the outputs of the sister network embeddings. The max function takes the largest value of 0 and the margin, m, minus the distance.

Actually it tries to distance be zero if pairs are the similar and be margin if dissimilar.

![image](https://user-images.githubusercontent.com/47561760/193100817-28d1ef81-5b87-4dda-919e-c7399bf8338c.png)

# Trainer Class
This class Does the main part of code which is training model, plot the training process and save model each n epochs.

I Defined `Contrastive` Loss and `Adam` Optimizer with learning rate 0.001 and 0.999 momentum.

Does Each training step in `train_step` function and validation step in `val_step` function and whole trining process in 
`train` function.
 
## Some Configurations
 
*   You can set epoch size : `EPOCHS` and batch size : `BATCH_SIZE`.
*   Set `device` that you want to train model on it : `device`(default runs on cuda if it's available)
*   You can set one of three `verboses` that prints info you want => 0 == nothing || 1 == model architecture || 2 == print optimizer || 3 == model parameters size.
*   Each time you train model weights and plot(if `save_plots` == True) will be saved in `save_dir`.
*   You can find a `configs` file in `save_dir` that contains some information about run. 



