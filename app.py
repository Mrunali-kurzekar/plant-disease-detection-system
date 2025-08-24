from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os
from dotenv import load_dotenv
import google.generativeai as genai

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file")
    exit(1)
genai.configure(api_key=GEMINI_API_KEY)

# Patch the standardize_reduction function to handle 'auto'
try:
    from tensorflow.keras.losses import Reduction, Loss

    def patched_standardize_reduction(reduction):
        if reduction == "auto":
            return Reduction.SUM_OVER_BATCH_SIZE
        if isinstance(reduction, str):
            return Reduction(reduction)
        return reduction

    Loss.standardize_reduction = staticmethod(patched_standardize_reduction)
except ImportError:
    print("Could not patch standardize_reduction. Please check your TensorFlow version.")
    exit(1)

# Define model paths and class names for each plant
MODELS = {
    "apple": {
        "path": "Models/Apple.h5",
        "classes": ["Healthy", "Scab", "Black Rot", "Cedar Rust"]
    },
    "cherry": {
        "path": "Models/Cherry.h5",
        "classes": ["Powdery Mildew", "Healthy"]
    },
    "corn": {
        "path": "Models/Corn.h5",
        "classes": ["Gray Leaf Spot", "Common Rust", "Northern Leaf Blight", "Healthy"]
    },
    "grape": {
        "path": "Models/Grape.h5",
        "classes": ["Black Rot", "Esca", "Leaf Blight", "Healthy"]
    },
    "peach": {
        "path": "Models/Peach.h5",
        "classes": ["Bacterial Spot", "Healthy"]
    },
    "pepper": {
        "path": "Models/Pepper.h5",
        "classes": ["Bacterial Spot", "Healthy"]
    },
    "potato": {
        "path": "Models/Potato.h5",
        "classes": ["Early Blight", "Late Blight", "Healthy"]
    },
    "strawberry": {
        "path": "Models/Strawberry.h5",
        "classes": ["Leaf Scorch", "Healthy"]
    }
}

# Load all models
loaded_models = {}
for plant, info in MODELS.items():
    try:
        print(f"Loading model for {plant}...")
        model = tf.keras.models.load_model(info["path"], compile=False)
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(reduction="sum_over_batch_size"),
            metrics=["accuracy"]
        )
        loaded_models[plant] = model
        print(f"Model for {plant} loaded successfully!")
    except Exception as e:
        print(f"Failed to load model for {plant}: {e}")
        exit(1)

def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/detector')
def detector():
    return render_template('index.html', plants=MODELS.keys())

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    if 'plant' not in request.form:
        return jsonify({'error': 'Plant type not specified'}), 400

    file = request.files['file']
    plant = request.form['plant'].lower()

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if plant not in loaded_models:
        return jsonify({'error': f"Invalid plant type: {plant}"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_img = preprocess_image(image)
        
        model = loaded_models[plant]
        prediction = model.predict(processed_img)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = MODELS[plant]["classes"][predicted_class_idx]
        confidence = float(prediction[0][predicted_class_idx])
        
        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'all_predictions': dict(zip(MODELS[plant]["classes"], prediction[0].tolist()))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ai-assistance', methods=['POST'])
def ai_assistance():
    try:
        data = request.get_json()
        user_query = data.get('query')

        if not user_query:
            return jsonify({'error': 'No query provided'}), 400

        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Prepare prompt
        prompt = f"You are AgriBot, an agriculture expert assistant. Provide accurate, concise, and practical advice on plant diseases, crop care, soil management, or related agricultural topics. Focus on actionable steps and avoid speculative information. User's query: {user_query}"

        # Generate response
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 500
            }
        )

        # Extract text response
        gemini_response = response.text

        return jsonify({'response': gemini_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    app.run(debug=True, host='0.0.0.0', port=5000)