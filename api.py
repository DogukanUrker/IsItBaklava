from flask import Flask, request, jsonify  # Import necessary modules from Flask
from test import (
    predictImageFromBytes,
)  # Import the function to predict images from bytes
import requests  # Import requests module to handle HTTP requests

# Initialize the Flask app
app = Flask(__name__)


# Define a route for the API to accept image uploads
@app.route("/predict", methods=["POST"])  # Define the route and the HTTP method
def predict():
    if "image" not in request.files:  # Check if 'image' is not in the uploaded files
        return (
            jsonify({"error": "No image provided"}),
            400,
        )  # Return an error response if no image is provided

    file = request.files["image"]  # Get the uploaded image file
    try:
        imageBytes = file.read()  # Read the image file as bytes
        result = predictImageFromBytes(
            imageBytes
        )  # Predict the image using the imported function
        return jsonify(
            {"result": result}
        )  # Return the prediction result as a JSON response
    except Exception as e:  # Catch any exception that occurs
        return (
            jsonify({"error": str(e)}),
            500,
        )  # Return an error response with the exception message


# Define a route for the API to accept image URLs
@app.route("/predict-url", methods=["POST"])  # Define the route and the HTTP method
def predict_url():
    if "url" not in request.json:  # Check if 'url' is not in the JSON payload
        return (
            jsonify({"error": "No URL provided"}),
            400,
        )  # Return an error response if no URL is provided

    url = request.json["url"]  # Get the URL from the JSON payload
    try:
        response = requests.get(url)  # Send a GET request to the URL
        response.raise_for_status()  # Raise an HTTPError for bad responses
        imageBytes = response.content  # Get the content of the response (image bytes)
        result = predictImageFromBytes(
            imageBytes
        )  # Predict the image using the imported function
        return jsonify(
            {"result": result}
        )  # Return the prediction result as a JSON response
    except (
        requests.exceptions.RequestException
    ) as e:  # Catch any request-related exceptions
        return (
            jsonify({"error": str(e)}),
            500,
        )  # Return an error response with the exception message
    except Exception as e:  # Catch any other exceptions that occur
        return (
            jsonify({"error": str(e)}),
            500,
        )  # Return an error response with the exception message


# Run the app
if __name__ == "__main__":  # Check if the script is being run directly (not imported)
    app.run(debug=True)  # Run the Flask app in debug mode
