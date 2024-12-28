from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from werkzeug.utils import secure_filename

# Initialize Flask application
app = Flask(__name__)

# Data paths
DATA_FOLDER = 'data'
IMAGE_FOLDER = os.path.join(DATA_FOLDER, 'test')
CSV_PATH = os.path.join(DATA_FOLDER, 'test.csv')
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
UPLOAD_FOLDER = os.path.join(DATA_FOLDER, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load data
data = pd.read_csv(CSV_PATH).fillna('N/A')  # Replace NaN with 'N/A'

# Model paths
MODEL_PATHS = {
    "cnn": "models/best_cnn_model_traced.pt",
    "efficientnet": "models/best_efficientnet_model_traced.pt",
    "cnn_metadata": "models/best_cnn_metadata_model_traced.pt"
}

# Load models
device = torch.device("cpu")
models = {key: torch.jit.load(path, map_location=device).eval() for key, path in MODEL_PATHS.items()}

# Image preprocessing
IMAGE_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Metadata processing
def process_metadata(age, sex, anatom_site):
    sex_categories = ['male', 'female']
    anatom_categories = ['torso', 'lower extremity', 'upper extremity', 'head/neck', 'palms/soles', 'oral/genital']

    def _one_hot_encode(value, categories):
        encoding = [0.0] * len(categories)
        if value in categories:
            encoding[categories.index(value)] = 1.0
        return encoding

    metadata_values = _one_hot_encode(sex, sex_categories)
    metadata_values += _one_hot_encode(anatom_site, anatom_categories)
    metadata_values.append(float(age))

    return torch.tensor(metadata_values, dtype=torch.float32).unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html', columns=data.columns)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    model_type = request.form.get('model_type', 'cnn')

    # Safely parse the age input
    age_input = request.form.get('age', '')
    try:
        age = int(age_input) if age_input else 40  # Default to 40 if empty
    except ValueError:
        return jsonify({'error': 'Invalid age value. Please enter a valid number.'}), 400

    sex = request.form.get('sex', 'male')
    anatom_site = request.form.get('anatom_site', 'torso')

    try:
        image = Image.open(filepath).convert('RGB')
        input_image = transform(image).unsqueeze(0)

        model = models.get(model_type, models['cnn'])
        input_image = input_image.to(device)

        if model_type == 'cnn_metadata':
            metadata = process_metadata(age, sex, anatom_site).to(device)
            with torch.no_grad():
                output = model(input_image, metadata)
        else:
            with torch.no_grad():
                output = model(input_image)

        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        class_names = ["Benign", "Malignant"]
        predicted_label = class_names[predicted_class]

        # Set alert class based on prediction
        alert_class = "alert-success" if predicted_label == "Benign" else "alert-danger"

        result = {
            "predicted_class": predicted_label,
            "confidence": float(confidence),
            "alert_class": alert_class,
            "probabilities": {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/browse', methods=['GET'])
def browse():
    # Get query parameters for pagination and search
    query = request.args.get('search[value]', '').lower()
    draw = int(request.args.get('draw', 1))
    start = int(request.args.get('start', 0))
    length = int(request.args.get('length', 10))

    # Filter data if search query is provided
    filtered_data = data
    if query:
        filtered_data = data[data.apply(lambda row: query in str(row).lower(), axis=1)]

    # Total records
    total_records = len(data)
    total_filtered = len(filtered_data)

    # Paginate the filtered data
    paginated_data = filtered_data.iloc[start:start + length]
    response = {
        "draw": draw,
        "recordsTotal": total_records,
        "recordsFiltered": total_filtered,
        "data": [
            {
                **row.to_dict(),
                "image_view": f'<button class="btn" onclick="viewImage(\'{row["image_name"]}\')"><i class="fa fa-eye"></i></button>'
            }
            for _, row in paginated_data.iterrows()
        ]
    }
    return jsonify(response)


@app.route('/data/test/<filename>')
def serve_image(filename):
    image_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
