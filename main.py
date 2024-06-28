import os
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

# Define the directory where the dataset is stored and the class names
dataDir = './data'
classes = ['baklava', 'notBaklava']

# Define the target image size for resizing
imageSize = (224, 224)

# Define the data augmentation and preprocessing transformations
dataTransforms = transforms.Compose([
    transforms.RandomResizedCrop(224),       # Randomly crop and resize the image to 224x224
    transforms.RandomHorizontalFlip(),       # Randomly flip the image horizontally
    transforms.RandomRotation(10),           # Randomly rotate the image by up to 10 degrees
    transforms.ToTensor(),                   # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the image
])

# Function to load images and their corresponding labels
def loadImages(dataDir, classes, imageSize):
    images = []   # List to store the images and their labels
    for label, cls in enumerate(classes):  # Iterate through each class
        clsDir = os.path.join(dataDir, cls)  # Path to the class directory
        for filename in os.listdir(clsDir):  # Iterate through each file in the class directory
            filepath = os.path.join(clsDir, filename)  # Full path to the file
            if os.path.isfile(filepath):  # Check if it's a file
                img = Image.open(filepath).convert('RGB')  # Open the image and convert to RGB
                img = img.resize(imageSize)  # Resize the image
                images.append((img, label))  # Append the image and its label to the list
    return images

# Load the images and their labels
imageLabelPairs = loadImages(dataDir, classes, imageSize)

# Load a pre-trained ResNet model for feature extraction
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()  # Set the model to evaluation mode

# Remove the final classification layer from the model
featureExtractor = torch.nn.Sequential(*list(model.children())[:-1])

# Function to extract features from images using the pre-trained model
def extractFeatures(images):
    features = []  # List to store the extracted features
    labels = []    # List to store the labels
    for image, label in images:  # Iterate through each image and label pair
        image = dataTransforms(image).unsqueeze(0)  # Apply transformations and add a batch dimension
        with torch.no_grad():  # Disable gradient computation
            feature = featureExtractor(image).numpy().flatten()  # Extract features and flatten them
        features.append(feature)  # Append the features to the list
        labels.append(label)      # Append the label to the list
    return np.array(features), np.array(labels)

# Extract features and labels from the images
features, labels = extractFeatures(imageLabelPairs)

# Split the dataset into training and validation sets
XTrain, XVal, yTrain, yVal = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler and SVM classifier
pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))

# Train the classifier on the training set
pipeline.fit(XTrain, yTrain)

# Predict labels for the validation set
yPred = pipeline.predict(XVal)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(yVal, yPred)
print(f'Validation Accuracy: {accuracy:.4f}')

# Save the trained model to a file
joblib.dump(pipeline, 'baklavaClassifier.pkl')

# Load the model from the file (for demonstration purposes)
pipeline = joblib.load('baklavaClassifier.pkl')

# Predict and print the first 5 predictions on the validation set
print(pipeline.predict(XVal[:5]))