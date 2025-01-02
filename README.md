# Cancer Image Analysis and Detection

The notebook focuses on cancer image analysis and detection using a convolutional neural network (CNN) model. You can also access the notebook on [Google Colab](https://colab.research.google.com/drive/1d80ICJ1M1PzWZ3aYYn00tREXpg0mUVoQ#scrollTo=SufTLnIXmOpF). The dataset and saved model weights can be accessed in [Google Drive](https://drive.google.com/drive/u/0/folders/1yK2W4D9y3VdnIg5SoDeGAWD1e1rSAKGp). 

Below is a step-by-step outline of the main components in the notebook:

## Step 1: Importing Required Libraries
The code begins with the import of essential libraries like Keras, TensorFlow, OpenCV, NumPy, and Matplotlib for building, loading, and visualizing the model and images. The pre-trained model and dataset files are also downloaded using the gdown command.


## Step 2: Model Creation and Compilation
The CNN model is constructed as follows:

a. Input Layer: Takes a 500x500x3 image.
b. Convolutional Layers: Feature extraction using two convolutional layers with ReLU activation.
c. Pooling Layers: Reduces the dimensionality with MaxPooling layers.
d. Flatten and Dense Layers: Converts feature maps into a single vector and feeds them into dense layers.
e. Output Layer: Outputs a sigmoid-activated single neuron for binary classification (e.g., benign or malignant).

The model is compiled using the RMSprop optimizer and a binary cross-entropy loss function.

## Step 3: Loading the Pre-trained Model
The code uses the model.load_weights to load the pre-trained weights from a downloaded file (model.h5).

## Step 4: Image Preprocessing and Visualization
The uploaded image is read using OpenCV and resized to 500x500 pixels.
The tumor regions are detected by:
a. Converting the image to grayscale.
b. Gaussian Blur is applied to smooth the image.
c. Using Otsu's thresholding for binary segmentation.
d. Find contours to detect regions of interest and draw bounding boxes around them.

## Step 5: Image Class Prediction
The preprocessed image is reshaped to match the input dimensions required by the model (1, 500, 500, 3) and fed into the model to get predictions. The output class is determined based on the highest predicted probability.

## Step 6: Additional Image Classification
Another sample image is downloaded and processed similarly for classification using the same pre-trained model.

Detailed explanation of the code is also given in [Cancer Image Analysis Tutorial](Cancer_Image_Analysis_Tutorial.pdf)
