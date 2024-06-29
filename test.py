import os  # Import the os module for interacting with the operating system
from PIL import Image  # Import the Image module from PIL for image processing
import torch  # Import the torch module for working with PyTorch
import torchvision.models as models  # Import the models module from torchvision for pre-trained models
import torchvision.transforms as transforms  # Import the transforms module from torchvision for data transformations
import joblib  # Import the joblib module for model serialization
import io  # Import the io module for handling byte streams

# Define the target image size for resizing
imageSize = (256, 256)

# Define the data preprocessing transformations (without augmentation for testing)
dataTransforms = transforms.Compose(
    [
        transforms.Resize(imageSize),  # Resize the image to 256x256
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.256, 0.225]
        ),  # Normalize the image with specified mean and standard deviation
    ]
)

# Load a pre-trained ResNet model for feature extraction
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()  # Set the model to evaluation mode

# Remove the final classification layer from the model to use it as a feature extractor
featureExtractor = torch.nn.Sequential(*list(model.children())[:-1])


# Function to extract features from a single image using the pre-trained model
def extractFeatures(image):
    # Apply transformations and add a batch dimension
    image = dataTransforms(image).unsqueeze(0)
    with torch.no_grad():  # Disable gradient computation
        # Extract features and flatten them
        feature = featureExtractor(image).numpy().flatten()
    return feature  # Return the extracted features


# Load the trained model from the .pkl file
pipeline = joblib.load("baklavaClassifier.pkl")


# Function to predict if an image is baklava or not using the extracted features
def prediction(image):
    img = Image.open(image).convert("RGB")  # Open the image and convert to RGB
    feature = extractFeatures(img)  # Extract features from the image
    result = pipeline.predict([feature])  # Predict the class using the trained model
    return (
        True if result[0] == 0 else False
    )  # Return True if the class is baklava (0), otherwise False


# Function to predict if an image is baklava or not given its file path
def predictImage(imagePath):
    result = prediction(imagePath)  # Predict the image
    return result  # Return the prediction result


# Function to predict if an image is baklava or not given its byte content
def predictImageFromBytes(imageBytes):
    img = io.BytesIO(imageBytes)  # Convert the byte content to a byte stream
    result = prediction(img)  # Predict the image
    return result  # Return the prediction result


# Function to predict if images in a directory are baklava or not
def predictImagesInDirectory(directoryPath):
    results = []  # Initialize an empty list to store the results
    for filename in os.listdir(
        directoryPath
    ):  # Iterate over the files in the directory
        filepath = os.path.join(
            directoryPath, filename
        )  # Get the full path of the file
        if os.path.isfile(filepath):  # Check if it is a file
            result = predictImage(filepath)  # Predict the image
            results.append((filename, result))  # Append the result to the list
    return results  # Return the list of results
