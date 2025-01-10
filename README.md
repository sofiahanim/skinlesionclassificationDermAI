# DermAI Skin Lesion Classification

## Project Overview
DermAI is a Flask-based application designed to classify skin lesions using ML models that supports two model options which are CNN (image-only) and CNN + Metadata (image with metadata)

---

## Important Project Directory

```plaintext
├── app.py                           # Backend Logic/Model Logic
├── templates/
│   └── index.html                   # Web UI
├── data/
│   ├── test/                        # Directory for test images
│   └── test.csv                     # Metadata for test images
├── models/
│   ├── best_cnn_model_traced.pt            # CNN model
│   └── best_cnn_metadata_model_traced.pt   # CNN model with metadata
├── static/
│   ├── group.png                   
│   └── styles.css                   
├── Dockerfile                      
├── cloudbuild.yaml                  # Google Cloud Build configuration
├── requirements.txt                 # Python dependencies
└── README.md                    
```

---

## Requirements

- **Python 3.8+**
- **Flask 2.x**
- **PyTorch 2.0+**
- **Docker** (optional)
- **Google Cloud Build and Cloud Run** (for deployment)
- Dependencies listed in `requirements.txt`

---

## Installation and Usage

### Step 1: Clone the Repository
```bash
git clone https://github.com/sofiahanim/skinlesionclassificationDermAI.git
cd skinlesionclassificationDermAI
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python app.py
```
The application will be accessible at `http://127.0.0.1:5000`

---

## Features

- **Model Options**:
  - **CNN** provide basic classification using images only
  - **CNN + Metadata** enhanced classification with images and additional metadata such as age, sex, anatomical site

- **Predictions**:
  - Upload skin lesion images (PNG/JPG/JPEG)
  - Choose the model type and view predictions with confidence scores

- **Test Data Browser**:
  - Search and view test data in a table format
  - Preview and download test images

---

## Deployment to Google Cloud Run via Cloud Build

### Step 1: Configure Cloud Build in Google Cloud Console
1. Navigate to **Cloud Build** in the [Google Cloud Console](https://console.cloud.google.com/cloud-build)
2. Create a **new trigger**:
   - Connect GitHub repo
   - Set the trigger to build on branch `main`
   - Use `cloudbuild.yaml` file for build configuration

### Step 2: Deploy to Cloud Run
1. Once the build is complete, go to **Cloud Run** in the Google Cloud Console
2. Deploy container image created by Cloud Build

---

## cloudbuild.yaml

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/skin-lesion-classification', '.']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/skin-lesion-classification']

  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'skin-lesion-classification'
      - '--image'
      - 'gcr.io/$PROJECT_ID/skin-lesion-classification'
      - '--platform'
      - 'managed'
      - '--region'
      - 'us-central1'
      - '--allow-unauthenticated'
```

---

## Notes

- Update the `project-id` and `region` placeholders in `cloudbuild.yaml` with actual values
- The application will be accessible at URL provided eg: `https://skin-lesion-classification-[id].run.app`

---




