# DermAI Skin Lesion Classification

## Project Overview
DermAI is a Flask-based application designed to classify skin lesions using ML models. The application supports two model options: CNN (image-only) and CNN + Metadata (image with metadata).

---

## Project Directory Structure
```
├── app.py                   # Flask backend logic
├── templates/
│   └── index.html           # Web interface
├── data/
│   ├── test/                # Directory for test images
│   ├── test.csv             # Metadata for test images
├── models/
│   ├── best_cnn_model_traced.pt            # CNN model
│   └── best_cnn_metadata_model_traced.pt   # CNN model with metadata
├── Dockerfile               # Docker configuration file
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Requirements
- Python 3.8+
- Flask 2.x
- PyTorch 2.0+
- Docker (optional, for containerization)

Dependencies are listed in `requirements.txt`.

---

## Installation and Usage

### Step 1: Clone the Repository
```bash
git clone https://github.com/{repo}/skin-lesion-classification.git
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
The application will be accessible at `http://127.0.0.1:5000`.

---

## Features
- **Model Options**:
  - CNN: Basic classification using images only
  - CNN + Metadata: Enhanced classification with images and additional metadata (age, sex, anatomical site)
- **Predictions**:
  - Upload skin lesion images (PNG/JPG/JPEG).
  - Choose the model type and view predictions with confidence scores
- **Test Data Browser**:
  - Search and view test data in a table format
  - Preview and download test images

---

## Dockerization

### Step 1: Build Docker Image
```bash
docker build -t skin-lesion-classification .
```

### Step 2: Run Docker Container
```bash
docker run -p 5000:5000 skin-lesion-classification
```

---

## Deployment to Google Cloud Run

### Step 1: Build and Push Docker Image
```bash
gcloud builds submit --tag gcr.io/{project-id}/skin-lesion-classification
```

### Step 2: Deploy to Cloud Run
```bash
gcloud run deploy skin-lesion-classification \
  --image gcr.io/{project-id}/skin-lesion-classification \
  --platform managed \
  --region region \
  --allow-unauthenticated
```

---

## Notes
- Update `project-id` and `region` placeholders with actual values when deploying to Google Cloud.

---

## Acknowledgments
- **[Flask](https://flask.palletsprojects.com/)**: Web framework.
- **[PyTorch](https://pytorch.org/)**: Machine learning framework.
- **[Bootstrap](https://getbootstrap.com/)**: Responsive UI framework.
- **[Google Cloud Platform](https://cloud.google.com/)**: Deployment infrastructure.

---


