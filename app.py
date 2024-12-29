from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from werkzeug.utils import secure_filename
from io import BytesIO

# Initialize Flask application
app = Flask(__name__)

# Data paths
DATA_FOLDER = 'data'
IMAGE_FOLDER = os.path.join(DATA_FOLDER, 'test')
CSV_PATH = os.path.join(DATA_FOLDER, 'test.csv')

app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# Load data
data = pd.read_csv(CSV_PATH).fillna('N/A')  # Handle NaN values

# Model paths
MODEL_PATHS = {
    "cnn": "models/best_cnn_model_traced.pt",
    "efficientnet": "models/best_efficientnet_model_traced.pt",
    "cnn_metadata": "models/best_cnn_metadata_model_traced.pt"
}

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = {model_name: torch.jit.load(model_path, map_location=device).eval()
          for model_name, model_path in MODEL_PATHS.items()}


# Image preprocessing
IMAGE_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Explicit One-Hot Encoding
def process_metadata(age, sex, anatom_site):
    # Define the categories for sex and anatomical site
    sex_categories = ['male', 'female']
    anatom_categories = ['torso', 'lower extremity', 'upper extremity', 'head/neck', 'palms/soles', 'oral/genital']
    
    # Initialize the metadata list with zeros for each category
    metadata = [0] * (len(sex_categories) + len(anatom_categories))
    
    # Check and encode 'sex'
    if sex in sex_categories:
        metadata[sex_categories.index(sex)] = 1
    else:
        raise ValueError(f"Invalid 'sex' provided: {sex}. Expected one of {sex_categories}")
    
    # Check and encode 'anatom_site'
    if anatom_site in anatom_categories:
        metadata[len(sex_categories) + anatom_categories.index(anatom_site)] = 1
    else:
        raise ValueError(f"Invalid 'anatom_site' provided: {anatom_site}. Expected one of {anatom_categories}")
    
    # Append age as a float
    metadata.append(float(age))
    
    # Convert list to a PyTorch tensor with an additional dimension for batch size
    return torch.tensor([metadata], dtype=torch.float32)


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
        model = models.get(model_type, None)

        if model is None:
            raise ValueError(f"Model type '{model_type}' is not recognized.")

        # Process the prediction differently based on model type
        if model_type == 'cnn_metadata':
            age = request.form.get('age', 40, type=int)
            sex = request.form.get('sex', 'male')
            anatom_site = request.form.get('anatom_site', 'torso')
            metadata = process_metadata(age, sex, anatom_site)

            with torch.no_grad():
                output = model(input_image, metadata)
        else:
            with torch.no_grad():
                output = model(input_image)

        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_class_index = np.argmax(probabilities)
        predicted_class = ["Benign", "Malignant"][predicted_class_index]
        confidence = probabilities[predicted_class_index]

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        # Logging the exception can be helpful for debugging
        app.logger.error(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred during processing', 'details': str(e)}), 500

@app.route('/browse', methods=['GET'])
def browse():
    query = request.args.get('search[value]', '').lower()
    start = int(request.args.get('start', 0))
    length = int(request.args.get('length', 10))
    column_index = int(request.args.get('order[0][column]', 0))  # Default to the first column
    sort_direction = request.args.get('order[0][dir]', 'asc')  # Default sorting direction

    column_names = ['patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'width', 'height', 'image_name']
    sort_column = column_names[column_index]  # Ensure this matches the order in your DataTables initialization

    filtered_data = data[data.apply(lambda row: query in str(row).lower(), axis=1)]

    # Sort the data
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
    #filename = filename + '.jpg'  # Append the extension to the filename
    secure_path = os.path.join(app.config['IMAGE_FOLDER'], secure_filename(filename))
    print("Trying to serve:", secure_path)  # This will output the path it's trying to access
    if not os.path.exists(secure_path):
        return jsonify({'error': 'Image not found'}), 404
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
