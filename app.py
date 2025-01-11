from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)

# Data paths
DATA_FOLDER = 'data'
IMAGE_FOLDER = os.path.join(DATA_FOLDER, 'test')
CSV_PATH = os.path.join(DATA_FOLDER, 'test.csv')

app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

data = pd.read_csv(CSV_PATH).fillna('N/A')  # Handle NaN values

# Model paths
MODEL_PATHS = {
    "cnn": os.path.join("models", "best_cnn_model_traced.pt"),
    "cnn_metadata": os.path.join("models", "best_cnn_metadata_model_traced.pt")
}

# Image preprocessing
IMAGE_SIZE = 128
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {}  # Initialize an empty dictionary for models

def load_traced_model(model_path, device):
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

@app.route('/group.png')
def serve_group_image():
    return send_from_directory('static', 'group.png') 

@app.route('/styles.css')
def serve_css():
    return send_from_directory('static', 'styles.css') 

# Function to load models
def load_all_models():
    global models
    for model_name, model_path in MODEL_PATHS.items():
        models[model_name] = load_traced_model(model_path, device)
    print("Models loaded successfully!")


# Explicit One-Hot Encoding
def process_metadata(age, sex, anatom_site):
    sex_categories = ['male', 'female']
    anatom_categories = ['torso', 'lower extremity', 'upper extremity', 'head/neck', 'palms/soles', 'oral/genital']

    def _one_hot_encode(value, categories):
      encoding = np.zeros(len(categories), dtype=np.float32)
      if value in categories:
          encoding[categories.index(value)] = 1.0
      return encoding

    metadata_values = np.concatenate([
        _one_hot_encode(sex, sex_categories),
        _one_hot_encode(anatom_site, anatom_categories),
        np.array([float(age)], dtype=np.float32)
    ])
    metadata = torch.tensor(metadata_values, dtype=torch.float32)  # Return tensor
    return metadata

@app.route('/')
def index():
    return render_template('index.html', columns=data.columns.tolist())

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or no file uploaded'}), 400

    try:
        image = Image.open(BytesIO(file.read())).convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)
        model_type = request.form.get('model_type', 'cnn')
        model = models.get(model_type)

        if model is None:
            raise ValueError(f"Model type '{model_type}' is not recognized.")

        if model_type == 'cnn_metadata':
            age = request.form.get('age', 40, type=int)
            sex = request.form.get('sex', 'male')
            anatom_site = request.form.get('anatom_site', 'torso')
            metadata = process_metadata(age, sex, anatom_site).unsqueeze(0).to(device)  # Move metadata to device

            with torch.no_grad():
                output = model(input_image, metadata)  # Pass both image and metadata
        else:
             with torch.no_grad():
                output = model(input_image) # pass the image

        # 0 = Benign
        # 1 = Malignant
        # softmax compute probability for each class 
        # eg benign : probabilities = [0.88,0.12] -> non-cancerous
        # eg malignant : probabilities = [0.12,0.88] -> cancerous
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_class_index = np.argmax(probabilities)
        predicted_class = ["Benign", "Malignant"][predicted_class_index]
        confidence = float(probabilities[predicted_class_index])  
        probabilities_list = probabilities.tolist()

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities_list, 
            'probabilities_benign': probabilities_list[0],
            'probabilities_malignant': probabilities_list[1]

        })
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred during processing', 'details': str(e)}), 500

@app.route('/browse', methods=['GET'])
def browse():
    query = request.args.get('search[value]', '').lower()
    start = int(request.args.get('start', 0))
    length = int(request.args.get('length', 10))
    column_index = int(request.args.get('order[0][column]', 0))  
    sort_direction = request.args.get('order[0][dir]', 'asc')

    column_names = ['patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'width', 'height', 'image_name']
    sort_column = column_names[column_index]  

    filtered_data = data[data.apply(lambda row: query in str(row).lower(), axis=1)]

    if sort_direction == 'asc':
        filtered_data = filtered_data.sort_values(by=sort_column, ascending=True)
    else:
        filtered_data = filtered_data.sort_values(by=sort_column, ascending=False)

    paginated_data = filtered_data.iloc[start:start + length]

    response_data = [{
        **row.to_dict(),
        "image_view": f'<button class="btn" data-image-name="{row["image_name"]}"><i class="fa fa-eye"></i></button>'
    } for index, row in paginated_data.iterrows()]

    response = {
        "draw": int(request.args.get('draw', 1)),
        "recordsTotal": len(data),
        "recordsFiltered": len(filtered_data),
        "data": response_data
    }
    return jsonify(response)

@app.route('/data/test/<filename>')
def serve_image(filename):
    secure_path = os.path.join(app.config['IMAGE_FOLDER'], secure_filename(filename))
    print("Trying to serve:", secure_path) 
    if not os.path.exists(secure_path):
        return jsonify({'error': 'Image not found'}), 404
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

if __name__ == '__main__':
    load_all_models()
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
