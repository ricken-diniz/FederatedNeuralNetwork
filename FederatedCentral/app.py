from flask import Flask, request
from nn.federated_train import merge

def create_app():
    app = Flask(__name__)

    @app.route("/uploadModel", methods=["POST"])
    def upload_models():
        if "file" not in request.files:
            return "Doesnt exist file", 400

        file = request.files["file"]

        if file.filename == "":
            return "Empty file name", 400

        try:
        
            file.save("static_model.pt")

            if merge("federated_model.pt", "static_model.pt"):
                return f"Model merged", 200
            
            return "Fail in merge", 400
        
        except Exception as e:
            return f"Fail in save model \nException: {e}", 400

    return app

app = create_app()