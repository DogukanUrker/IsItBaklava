import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import joblib

# Define the target image size for resizing
imageSize = (256, 256)

# Define the data preprocessing transformations (without augmentation for testing)
dataTransforms = transforms.Compose(
    [
        transforms.Resize(imageSize),  # Resize the image to 256x256
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.256, 0.225]
        ),  # Normalize the image
    ]
)

# Load a pre-trained ResNet model for feature extraction
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()  # Set the model to evaluation mode

# Remove the final classification layer from the model
featureExtractor = torch.nn.Sequential(*list(model.children())[:-1])

# Function to extract features from a single image using the pre-trained model
def extractFeatures(image):
    image = dataTransforms(image).unsqueeze(
        0
    )  # Apply transformations and add a batch dimension
    with torch.no_grad():  # Disable gradient computation
        feature = (
            featureExtractor(image).numpy().flatten()
        )  # Extract features and flatten them
    return feature

# Load the trained model from the .pkl file
pipeline = joblib.load("baklavaClassifier.pkl")

# Function to predict if an image is baklava or not
def predictImage(imagePath):
    img = Image.open(imagePath).convert("RGB")  # Open the image and convert to RGB
    feature = extractFeatures(img)  # Extract features from the image
    prediction = pipeline.predict([feature])  # Predict the class
    return True if prediction[0] == 0 else False # Return True if the prediction is 0 (baklava), False otherwise

def predictImagesInDirectory(directoryPath):
    results = []
    for filename in os.listdir(directoryPath):
        filepath = os.path.join(directoryPath, filename)
        if os.path.isfile(filepath):
            result = predictImage(filepath)
            results.append((filename, result))
    return results

""" # Test the function with an example image
imagePath = "./data/test/1.jpg"  # Replace with the path to your test image
result = predictImage(imagePath)
print(result)
 """
 
# Test the function with all images in a directory
directoryPath = "./data/test"  # Test images directory
predictions = predictImagesInDirectory(directoryPath)

# Print the results
for filename, result in predictions:
    print(f"The image {filename} is: {result}")