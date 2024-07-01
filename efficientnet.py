# Import necessary libraries
import os  # For operating system related functionalities
import numpy as np  # Numerical operations library
from PIL import Image  # Python Imaging Library for image processing
import torch  # PyTorch deep learning framework
import torchvision.transforms as transforms  # Data transformations for images
from sklearn.model_selection import train_test_split  # Splitting dataset
from sklearn.preprocessing import (
    StandardScaler,
)  # Standardize features by removing the mean and scaling to unit variance
from sklearn.svm import SVC  # Support Vector Classifier from scikit-learn
from sklearn.pipeline import (
    make_pipeline,
)  # Constructing a pipeline from transformers and estimator
from sklearn.metrics import (
    accuracy_score,
)  # Evaluation metric for classification accuracy
import joblib  # Save and load Python objects (including sklearn models) to and from disk
import time  # Time-related functionalities
import ssl  # SSL certificate handling
import certifi  # Mozilla's CA Bundle
from efficientnet_pytorch import EfficientNet  # EfficientNet model

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context

print("\n")  # Print newline for better readability

appStartTime = time.time()  # Record start time for app

# Define the directory where the dataset is stored and the class names
dataDir = "./data"  # Directory containing the dataset
classes = ["baklava", "notBaklava"]  # Class names for classification

# Define the target image size for resizing
imageSize = (256, 256)

# Define the data augmentation and preprocessing transformations
dataTransforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            256
        ),  # Crop the image to random size and aspect ratio
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(
            10
        ),  # Rotate the image by a random angle within [-10, 10] degrees
        transforms.ToTensor(),  # Convert the image to PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize image with specific mean and std
    ]
)


# Function to load images and their corresponding labels
def loadImages(dataDir, classes, imageSize):
    images = []  # List to store images and their labels
    startTime = time.time()  # Record start time for loading images
    print(f"Loading images from '{dataDir}'...")  # Print progress message
    for label, cls in enumerate(classes):  # Iterate through each class
        clsDir = os.path.join(dataDir, cls)  # Path to class directory
        classImages = os.listdir(clsDir)  # List of images in the class directory
        print(
            f"Found {len(classImages)} images in class '{cls}'"
        )  # Print number of images found
        for filename in classImages:  # Iterate through each image in the class
            filepath = os.path.join(clsDir, filename)  # Full path to the image file
            if os.path.isfile(filepath):  # Check if it's a file (not a directory)
                img = Image.open(filepath).convert(
                    "RGB"
                )  # Open image and convert to RGB mode
                img = img.resize(imageSize)  # Resize image to predefined size
                images.append((img, label))  # Append tuple of image and its label
    endTime = time.time()  # Record end time for loading images
    print(
        f"Loaded {len(images)} images in {(endTime - startTime):.2f} seconds."
    )  # Print loading time
    return images  # Return list of loaded images and labels


# Load the images and their labels
imageLabelPairs = loadImages(dataDir, classes, imageSize)  # Call loadImages function

print("\n")  # Print newline for better readability

# Load a pre-trained EfficientNet-B0 model for feature extraction
startTime = time.time()  # Record start time for loading EfficientNet model
print("Loading pre-trained EfficientNet-B0 model...")  # Print progress message
model = EfficientNet.from_pretrained(
    "efficientnet-b0"
)  # Load pre-trained EfficientNet-B0 model
model.eval()  # Set model to evaluation mode (not training mode)
featureExtractor = (
    model.extract_features
)  # Use the extract_features method for feature extraction
endTime = time.time()  # Record end time for loading EfficientNet model
print(
    f"Loaded EfficientNet-B0 model in {(endTime - startTime):.2f} seconds."
)  # Print loading time

print("\n")  # Print newline for better readability


# Function to extract features from images using the pre-trained model
def extractFeatures(images):
    features = []  # List to store extracted features
    labels = []  # List to store corresponding labels
    print(f"Extracting features from {len(images)} images...")  # Print progress message
    for idx, (image, label) in enumerate(
        images, start=1
    ):  # Iterate through each image-label pair
        image = dataTransforms(image).unsqueeze(
            0
        )  # Apply data transformations and add batch dimension
        with torch.no_grad():  # Disable gradient calculation
            feature = (
                featureExtractor(image).numpy().flatten()
            )  # Extract features and convert to numpy array
        features.append(feature)  # Append extracted feature to list
        labels.append(label)  # Append corresponding label to list
        if idx % 1000 == 0 or idx == len(
            images
        ):  # Print progress every 1000 images processed or at the end
            print(f"Processed {idx}/{len(images)} images...")  # Print progress message
    return np.array(features), np.array(
        labels
    )  # Convert lists to numpy arrays and return


# Extract features and labels from the images
startTime = time.time()  # Record start time for feature extraction
features, labels = extractFeatures(imageLabelPairs)  # Call extractFeatures function
endTime = time.time()  # Record end time for feature extraction
print(
    f"Extracted features and labels in {(endTime - startTime):.2f} seconds."
)  # Print extraction time

print("\n")  # Print newline for better readability

# Split the dataset into training and validation sets
startTime = time.time()  # Record start time for dataset splitting
print("Splitting dataset into train and validation sets...")  # Print progress message
XTrain, XVal, yTrain, yVal = train_test_split(
    features, labels, test_size=0.2, random_state=42
)  # Split dataset
endTime = time.time()  # Record end time for dataset splitting
print(
    f"Split dataset into train and validation sets in {(endTime - startTime):.2f} seconds."
)  # Print splitting time

print("\n")  # Print newline for better readability

# Create a pipeline with StandardScaler and SVM classifier
startTime = time.time()  # Record start time for pipeline creation
print(
    "Creating pipeline with StandardScaler and SVM classifier..."
)  # Print progress message
pipeline = make_pipeline(
    StandardScaler(), SVC(kernel="linear", probability=True)
)  # Create pipeline
endTime = time.time()  # Record end time for pipeline creation
print(
    f"Created pipeline in {(endTime - startTime):.2f} seconds."
)  # Print pipeline creation time

print("\n")  # Print newline for better readability

# Train the classifier on the training set
startTime = time.time()  # Record start time for training
print("Training SVM classifier...")  # Print progress message
pipeline.fit(XTrain, yTrain)  # Train pipeline on training set
endTime = time.time()  # Record end time for training
print(
    f"Trained classifier in {(endTime - startTime):.2f} seconds."
)  # Print training time

print("\n")  # Print newline for better readability

# Predict labels for the validation set
startTime = time.time()  # Record start time for prediction
print("Predicting labels for validation set...")  # Print progress message
yPred = pipeline.predict(XVal)  # Predict labels for validation set
endTime = time.time()  # Record end time for prediction
print(
    f"Predicted labels for validation set in {(endTime - startTime):.2f} seconds."
)  # Print prediction time

print("\n")  # Print newline for better readability

# Calculate the accuracy of the classifier
startTime = time.time()  # Record start time for accuracy calculation
accuracy = accuracy_score(yVal, yPred)  # Calculate accuracy
endTime = time.time()  # Record end time for accuracy calculation
print(
    f"Calculated accuracy in {(endTime - startTime):.2f} seconds."
)  # Print accuracy calculation time
print(f"Validation Accuracy: {accuracy:.4f}")  # Print validation accuracy

print("\n")  # Print newline for better readability

# Save the trained model to a file
startTime = time.time()  # Record start time for model saving
print("Saving trained model to file...")  # Print progress message
joblib.dump(pipeline, "baklavaClassifier.pkl")  # Save pipeline to file
endTime = time.time()  # Record end time for model saving
print(
    f"Saved trained model to file in {(endTime - startTime):.2f} seconds."
)  # Print model saving time

print("\n")  # Print newline for better readability

# Load the model from the file (for demonstration purposes)
startTime = time.time()  # Record start time for model loading
print("Loading model from file...")  # Print progress message
pipeline = joblib.load("baklavaClassifier.pkl")  # Load pipeline from file
endTime = time.time()  # Record end time for model loading
print(
    f"Loaded model from file in {(endTime - startTime):.2f} seconds."
)  # Print model loading time

print("\n")  # Print newline for better readability

# Predict and print the first 10 predictions on the validation set
startTime = time.time()  # Record start time for prediction
print("Predicting first 10 samples from validation set…")  # Print progress message
print(pipeline.predict(XVal[:10]))  # Predict and print first 10 samples
endTime = time.time()  # Record end time for prediction
print(
    f"Predicted first 10 samples in {(endTime - startTime):.2f} seconds."
)  # Print prediction time

print("\n")  # Print newline for better readability

appEndTime = time.time()  # Record end time for app
print(
    f"Training and testing completed successfully in {(appEndTime - appStartTime):.2f} seconds."
)  # Print completion message

print("\n")  # Print newline for better readability
