# Cancer_Image_Analysis-Detection

The notebook focus on cancer image analysis and detection using a convolutional neural network (CNN) model. Below is a step-by-step outline of the main components in the notebook:

# Step 1: Importing Required Libraries
The code begins with the import of essential libraries like Keras, TensorFlow, OpenCV, NumPy, and Matplotlib for building, loading, and visualizing the model and images. Additionally, the pretrained model and dataset files are downloaded using the gdown command.


# Step 2: Model Creation and Compilation
The CNN model is constructed as follows:

a. Input Layer: Takes a 500x500x3 image.
b. Convolutional Layers: Feature extraction using two convolutional layers with ReLU activation.
c. Pooling Layers: Reduces the dimensionality with MaxPooling layers.
d. Flatten and Dense Layers: Converts feature maps into a single vector and feeds them into dense layers.
e. Output Layer: Outputs a sigmoid-activated single neuron for binary classification (e.g., benign or malignant).

The model is compiled using the RMSprop optimizer and a binary cross-entropy loss function.

# Step 3: Loading the Pretrained Model
The code uses model.load_weights to load the pretrained weights from a downloaded file (model.h5).

# Step 4: Image Preprocessing and Visualization
The uploaded image is read using OpenCV and resized to 500x500 pixels.
The tumor regions are detected by:
a. Converting the image to grayscale.
b. Applying Gaussian Blur to smooth the image.
c. Using Otsu's thresholding for binary segmentation.
d. Finding contours to detect regions of interest and drawing bounding boxes around them.

# Step 5: Image Prediction
The preprocessed image is reshaped to match the input dimensions required by the model (1, 500, 500, 3) and fed into the model to get predictions. The output class is determined based on the highest predicted probability.

# Step 6: Additional Image Classification
Another sample image is downloaded and processed similarly for classification using the same pretrained model.
