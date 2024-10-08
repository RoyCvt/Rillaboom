from flask import Flask, jsonify, request
from flask_cors import CORS

from image_registration import register_images

app = Flask(__name__)

# Allow CORS
CORS(app)

# Define the endpoints accessible to the client
accessible_endpoints = ["index", "perform_registration"]


@app.route("/")
@app.route("/index")
def index():
    return jsonify({"message": "Server is working!", "status": 200}), 200


@app.route("/perform-registration", methods=["POST"])
def perform_registration():
    try:
        data = request.get_json()

        # Extract base64 encoded images from request data
        base64_image1 = data.get("base64Image1")
        base64_image1 = base64_image1.replace('data:image/jpeg;base64,', '') ###
        base64_image2 = data.get("base64Image2")
        base64_image2 = base64_image2.replace('data:image/jpeg;base64,', '') ###

        # Perform registration using a separate python script
        registration_result = register_images(base64_image1, base64_image2)
        registration_successful = registration_result[0]
        if registration_successful:
            base64_registered_image1, base64_registered_image2 = registration_result[1:]
            base64_registered_image1 = 'data:image/jpeg;base64,' + base64_registered_image1 ###
            base64_registered_image2 = 'data:image/jpeg;base64,' + base64_registered_image2 ###
            return jsonify({"images": [base64_registered_image1, base64_registered_image2], "status": 200}), 200
        else:
            failure_reason, status_code = registration_result[1:]
            return jsonify({"message": failure_reason, "status": status_code}), status_code 

    except Exception as e:
        return jsonify({"message": f"Internal Server Error: {e}", "status": 500}), 500


@app.before_request
def before_request():
    if request.endpoint not in accessible_endpoints:
        return jsonify({"message": "Resource not found!", "status": 404}), 404


def run_server():
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    run_server()
