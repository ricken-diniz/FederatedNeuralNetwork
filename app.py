from flask import Flask, request
from nn.federated_train import merge
import os

def create_app():
    app = Flask(__name__)

    @app.route("/uploadModel", methods=["POST"])
    def upload_models():
        if "file" not in request.files:
            return "Nenhum arquivo enviado", 400

        file = request.files["file"]

        if file.filename == "":
            return "Nome de arquivo vazio", 400

        
        if file.save("static_model.pt"):

            if merge("federated_model.pt", "static_model.pt"):
                return f"Modelo recebido e mesclado", 200
            
            return "Falha na mesclagem", 400
        
        return "Falha em salvar o modelo recebido", 400

    return app

app = create_app()