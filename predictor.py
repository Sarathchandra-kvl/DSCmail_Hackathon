from flask import Flask, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
from threading import Lock
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 512))
PORT = int(os.getenv("PORT", 5000))

# Load model and tokenizer once
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model_lock = Lock()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or "text" not in data:
            return jsonify({"error": "Invalid input. Provide a 'text' field."}), 400
        
        text = data["text"].strip()
        if len(text) == 0:
            return jsonify({"error": "Input text cannot be empty."}), 400
        if len(text) > MAX_INPUT_LENGTH:
            return jsonify({"error": f"Input exceeds max length of {MAX_INPUT_LENGTH}."}), 400

        # Optional parameters from request
        max_new_tokens = data.get("max_new_tokens", 3)

        with model_lock:
            input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(device)
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                do_sample=True,
                no_repeat_ngram_size=2
            )
        
        predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Generated text for input: {text}")
        return jsonify({"prediction": predicted_text})
    
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        return jsonify({"error": "Invalid input format"}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    from waitress import serve
    logger.info(f"Starting server on port {PORT}")
    serve(app, host="0.0.0.0", port=PORT)
