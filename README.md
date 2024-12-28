# DermAI Skin Lesion Classification

This project is a Flask-based web application for classifying skin lesions using machine learning models. Users can upload skin lesion images and optionally provide metadata (age, sex, anatomical site) for improved predictions.

---

## Project Structure

```plaintext
├── app.py                   # Main Flask application
├── templates/
│   └── index.html           # HTML template for the web app
├── data/
│   ├── test/                # Directory containing test images
│   ├── test.csv             # Metadata for the test images
├── models/
│   ├── best_cnn_model_traced.pt
│   ├── best_efficientnet_model_traced.pt
│   └── best_cnn_metadata_model_traced.pt
├── Dockerfile               # Docker configuration for containerization
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Requirements

- Python 3.8+
- Flask 2.x
- PyTorch 2.5.1
- Docker (for containerization)
- Google Cloud SDK (for Cloud Run deployment)

---

## Installation and Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/repo/skin-lesion-classification.git
cd skin-lesion-classification
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`.

---

## Dockerization

### Step 1: Build the Docker Image

```bash
docker build -t skin-lesion-classification .
```

### Step 2: Run the Docker Container

```bash
docker run -p 5000:5000 skin-lesion-classification
```

---

## Deploy to Google Cloud Run

### Step 1: Build and Push the Docker Image to Google Container Registry

```bash
gcloud builds submit --tag gcr.io/project-id/skin-lesion-classification
```

### Step 2: Deploy to Cloud Run

```bash
gcloud run deploy skin-lesion-classification \
  --image gcr.io/project-id/skin-lesion-classification \
  --platform managed \
  --region region \
  --allow-unauthenticated
```

---

## Access Logs from Docker Container

```bash
docker logs <container-id>
```

---

## Features

- **Model Options**:
  - CNN: Image-only classification.
  - EfficientNet: Image-only classification.
  - CNN + Metadata: Image classification with additional metadata (age, sex, anatomical site).
- **Interactive UI**: Upload images, choose models, and view predictions.
- **Table View**: Display and search test data.
- **Real-time Feedback**: Display prediction results with confidence scores.

---

## Acknowledgments

- **[Flask](https://flask.palletsprojects.com/)**: For the web framework.
- **[PyTorch](https://pytorch.org/)**: For building machine learning models.
- **[Bootstrap](https://getbootstrap.com/)**: For the responsive UI.
- **[Google Cloud Platform](https://cloud.google.com/)**: For deployment on Cloud Run.

---

## Notes

- Replace `repo` and `project-id` with the actual values for project.
- Ensure that the required models (`*.pt` files) and `test.csv` are correctly placed in their respective directories.
