import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

from pyngrok import ngrok
from flask_cors import CORS
from flask import Flask, request, send_file, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

@app.route('/image', methods=['GET'])
def generate_image():
    keywords = request.args.get('keywords')

    if not keywords:
        return jsonify({"error": "No keywords provided"}), 400

    prompt = f"a {keywords}, 8k, realistic photography"
    image = pipe(prompt).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return jsonify({"image": img_base64})

app.run(port=5000)