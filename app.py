from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "secret"
app.config['UPLOAD_FOLDER'] = 'data/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Load the model
model = tf.keras.models.load_model('skin_lesion_model (1).h5')

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if file part is in request
        if 'image' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['image']
        # Check if file is selected
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        # Check if file is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Save the file
            file.save(filepath)
            flash('File successfully uploaded', 'success')
            return redirect(url_for('predict', filename=filename))
        else:
            flash('Invalid file format. Please upload a .jpg, .jpeg, or .png file.', 'error')
    return render_template('upload.html')

@app.route('/predict/<filename>')
def predict(filename):
    try:
        # Build file path
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File {filename} does not exist in uploads.")

        # Load the image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))  # Resize to model input size
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Flatten the image if the model requires it
        image_array = image_array.reshape((image_array.shape[0], -1))

        # Debugging the shape of the input
        print(f"Input shape to the model: {image_array.shape}")

        # Make prediction
        predictions = model.predict(image_array)
        probabilities = predictions[0]

        # Define class labels
        labels = ['Actinic keratosis', 'Basal cell carcinoma', 'Benign keratosis',
                  'Dermatofibroma', 'Melanoma', 'Melanocytic nevus',
                  'Squamous cell carcinoma', 'Vascular lesion']
        probability_dict = dict(zip(labels, probabilities))

        # Generate a plot
        plt.figure(figsize=(10, 4))
        plt.bar(labels, probabilities, color='skyblue')
        plt.title('Prediction Probabilities')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')

    except FileNotFoundError:
        flash(f"File {filename} not found. Please upload again.", 'error')
        return redirect(url_for('upload_image'))
    except Exception as e:
        flash(f"An error occurred: {str(e)}", 'error')
        return redirect(url_for('upload_image'))

    return render_template('result.html', filename=filename, probabilities=probability_dict, plot_url=plot_url)

if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

