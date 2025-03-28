from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# 🔹 Load Pre-trained Spam Detection Model
loaded_model = joblib.load("/content/drive/MyDrive/spam_detector.pkl")

@app.route("/detect-spam", methods=["POST"])
def detect_spam():
    try:
        data = request.json
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No input text provided"}), 400

        # 🔹 Predict spam (1 = spam, 0 = not spam)
        prediction = loaded_model.predict([text])[0]
        
        return jsonify({"is_spam": bool(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5002, debug=True)
