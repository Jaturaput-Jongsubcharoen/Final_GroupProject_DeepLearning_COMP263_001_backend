# Deep Learning for Pneumonia Detection from Chest X-Ray Images — Backend

**Course:** COMP263-001 Deep Learning — Centennial College, Winter 2026

## Group Members

| # |            Name             |                  Role                  |                 File(s)                 |
|---|-----------------------------|----------------------------------------|-----------------------------------------|
| 1 | Drashtam Kinnar Banker      | Experiment 1 — Supervised Learning     | `first_exp.py` (Baseline CNN, Wide CNN) |
| 2 | Zeel Rameshbhai Vekariya    | Experiment 1 — Supervised Learning     | `first_exp.py` (Deep CNN)               |
| 3 | Jaturaput Jongsubcharoen    | Experiment 2 — Unsupervised Learning   | `second_exp.py`(Autoencoder)            |
| 4 | Juan David Barrero Guerrero | Experiment 2 — Unsupervised Learning   | `second_exp.py`(Autoencoder Transfer)   |
| 5 | Alim Rashyani               | Experiment 3 — State-of-the-Art Models | `third_exp.py` (ResNet50 From-Scratch)  |
| 6 | Raj Patel                   | Experiment 3 — State-of-the-Art Models | `third_exp.py` (ResNet50 Transfer)      |


## Backend Setup
---
## Project Description

This project applies deep learning to classify chest X-ray images as **Normal** or **Pneumonia**. Three experiments are conducted:

1. **Experiment 1 — Supervised Learning (Custom CNNs):** Three custom CNN architectures (Baseline, Deep, Wide) with varying depths and hyperparameters (`first_exp.py`).
2. **Experiment 2 — Unsupervised Learning (Autoencoder + Transfer):** A convolutional autoencoder is trained unsupervised; its encoder is transferred to a supervised classifier (`second_exp.py`).
3. **Experiment 3 — SOTA Models (ResNet50):** ResNet50 trained both from scratch and with ImageNet transfer learning (`third_exp.py`).

A **FastAPI backend** (`app.py`) serves all 6 trained models, metrics, and real-time predictions via REST API.

## Dataset

- **Name:** Chest X-Ray Images (Pneumonia)
- **Source:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **Classes:** 2 — Normal, Pneumonia
- **Split:** Train (80/20 train-val split), Test
- **Image Size:** 224 × 224 pixels (RGB)

### How to Download the Dataset

1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click **Download** (requires a free Kaggle account)
3. Extract the downloaded `archive.zip`
4. Place the extracted folder so the structure is:
   ```
   Final_GroupProject_DeepLearning_COMP263_001_backend/
   └── archive/
       └── chest_xray/
           ├── train/
           │   ├── NORMAL/
           │   └── PNEUMONIA/
           └── test/
               ├── NORMAL/
               └── PNEUMONIA/
   ```

## Prerequisites

- **Python:** 3.10.11
- **GPU (optional):** NVIDIA GPU with CUDA for faster training. CPU works but is slower.

## External Libraries / Dependencies

| Library | Purpose |
|---------|--------|
| tensorflow | Deep learning framework (model building, training, inference) |
| numpy | Numerical array operations |
| fastapi | REST API framework for serving models |
| uvicorn[standard] | ASGI server for FastAPI |
| python-multipart | Handling file uploads in FastAPI |
| Pillow | Image loading and preprocessing |
| scikit-learn | Evaluation metrics (precision, recall, F1, confusion matrix) |
| matplotlib | Generating training/validation plots |

All dependencies are listed in `requirements.txt`.

## How to Set Up and Run

### Step 1: Navigate to the Backend Folder

```bash
cd Final_GroupProject_DeepLearning_COMP263_001_backend
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Virtual Environment

**Windows (CMD):**
```bash
.venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
pip install scikit-learn matplotlib
```

### Step 5: Download and Place the Dataset

Follow the dataset instructions above. The `archive/chest_xray/` folder must be inside this backend directory.

### Step 6: Run the Training Experiments (Optional)

If trained model files already exist in `results_xray/`, training is skipped automatically. To retrain, delete the `.keras` files and run:

```bash
python first_exp.py
python second_exp.py
python third_exp.py
```

Each script will:
- Load and preprocess the dataset (80/20 train-val split)
- Train the model(s) with early stopping
- Evaluate on the test set (accuracy, precision, recall, F1-score)
- Save the trained model (`.keras`) and metrics (`.json`) to `results_xray/`
- Generate comparison plots (`.png`)

### Step 7: Start the API Server

```bash
uvicorn app:app --reload --port 8000
```

The server starts at **http://localhost:8000**.

- **API docs:** http://localhost:8000/docs
- **Models list:** http://localhost:8000/models
- **Model metrics:** http://localhost:8000/metrics/{model_name}
- **Prediction:** POST http://localhost:8000/predict (form: `image` file + `model_name` string)

## Project Structure

```
Final_GroupProject_DeepLearning_COMP263_001_backend/
├── app.py                  # FastAPI server (models, metrics, predictions)
├── first_exp.py            # Experiment 1: Custom CNNs (Baseline, Deep, Wide)
├── second_exp.py           # Experiment 2: Autoencoder + Transfer Learning
├── third_exp.py            # Experiment 3: ResNet50 (Transfer vs From-Scratch)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── archive/                # Dataset (download from Kaggle)
│   └── chest_xray/
└── results_xray/           # Trained models and metrics
    ├── exp1_custom_cnn/    # Experiment 1 outputs
    ├── exp2_autoencoder/   # Experiment 2 outputs
    └── exp3_*              # Experiment 3 outputs
```

## Trained Models (6 Total)

| # | Model Name | Experiment | File |
|---|-----------|-----------|------|
| 1 | Baseline CNN | Exp 1 — Supervised | `results_xray/exp1_custom_cnn/exp1_mri_baseline_cnn.keras` |
| 2 | Deep CNN | Exp 1 — Supervised | `results_xray/exp1_custom_cnn/exp1_mri_deep_cnn.keras` |
| 3 | Wide CNN | Exp 1 — Supervised | `results_xray/exp1_custom_cnn/exp1_mri_wide_cnn.keras` |
| 4 | Autoencoder Transfer | Exp 2 — Unsupervised | `results_xray/exp2_autoencoder/exp2_xray_transfer.keras` |
| 5 | ResNet50 Transfer | Exp 3 — SOTA | `results_xray/exp3_xray_resnet_transfer.keras` |
| 6 | ResNet50 From-Scratch | Exp 3 — SOTA | `results_xray/exp3_xray_resnet_scratch_FIXED.keras` |

## Frontend Setup
---
## Project Description

React single-page application for interacting with the trained deep learning models. Users can select a model, view its test-set metrics (accuracy, precision, recall, F1-score), upload a chest X-ray image, and receive a real-time Normal/Pneumonia prediction.

## Prerequisites

- **Node.js:** 20.x
- **npm:** Included with Node.js
- **Backend server** must be running on http://localhost:8000

## External Libraries / Dependencies

| Library | Purpose |
|---------|--------|
| react 18 | UI component framework |
| react-dom 18 | React DOM renderer |
| vite 5 | Development server and build tool |
| @vitejs/plugin-react 4 | Vite plugin for React JSX/HMR |
| eslint 9 | Code linting |

All dependencies are listed in `package.json`.

## How to Set Up and Run

### Step 1: Navigate to the Frontend Folder

```bash
cd Final_GroupProject_DeepLearning_COMP263_001_frontend
```

### Step 2: Install Dependencies

```bash
npm install
```

### Step 3: Start the Backend First

Open a **separate terminal** and start the backend API server (see Backend README for full instructions):

```bash
cd Final_GroupProject_DeepLearning_COMP263_001_backend
.venv\Scripts\activate
uvicorn app:app --reload --port 8000
```

### Step 4: Start the Frontend Development Server

```bash
npm run dev
```

Opens at **http://localhost:5173** in your browser.

### Step 5: Using the Application

1. **Select a Model** — Click one of the 6 model cards (Baseline CNN, Deep CNN, Wide CNN, Autoencoder Transfer, ResNet50 Transfer, ResNet50 From-Scratch). The selected card highlights with a pink border.
2. **View Metrics** — After selecting a model, the 4 gauge rings animate to show Accuracy, Precision, Recall, and F1-Score from the test set.
3. **Upload an Image** — Drag and drop a chest X-ray image onto the upload area, or click to browse files.
4. **Predict** — Click the **Predict** button. The app sends the image to the backend and displays the predicted class (Normal or Pneumonia), confidence percentage, and probability bars.
5. **Clear** — Click **Clear** to reset the image and result.

## Project Structure

```
Final_GroupProject_DeepLearning_COMP263_001_frontend/
├── index.html              # HTML entry point
├── package.json            # Node.js dependencies and scripts
├── vite.config.js          # Vite configuration
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── src/
│   ├── main.jsx            # React entry point
│   ├── App.jsx             # Main application component
│   ├── App.css             # Application styles
│   ├── index.css           # Global styles and CSS variables
│   └── components/
│       ├── ModelSelector.jsx   # Model card grid with icons
│       ├── MetricsPanel.jsx    # 4 metric gauge displays
│       ├── Gauge.jsx           # SVG ring gauge component
│       └── PredictPanel.jsx    # Image upload, predict button, results
```

## Build for Production

```bash
npm run build
```

Output is generated in the `dist/` folder.
