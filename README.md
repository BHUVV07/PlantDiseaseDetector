# Plant Disease Detector (Random Forest / SVM) - Scaffold

This project scaffold helps you build an AI-based plant disease detector using the PlantVillage dataset.
It contains training instructions, a Streamlit app for uploading a random leaf image, and utilities for preprocessing.

**Important:** The actual PlantVillage image dataset is large and cannot be bundled here. Follow the instructions below to download the dataset and place it in `sample_data/` (or point the scripts to your dataset path).

## Quick steps
1. Install requirements: `pip install -r requirements.txt`
2. Download the PlantVillage dataset (e.g., from Kaggle - https://www.kaggle.com/datasets/emmarex/plantdisease) and extract into `sample_data/PlantVillage/`
   - The expected structure: `sample_data/PlantVillage/<class_name>/*.jpg`
3. Train a model (example commands provided in `train_notebook.ipynb`)
4. Run the app: `streamlit run app_streamlit.py`
5. Open the Streamlit UI and upload a leaf image to get a prediction and disease information.

## What this project contains
- `app_streamlit.py` - Streamlit app to upload an image and show predicted class + disease info.
- `model_utils.py` - preprocessing utilities and example function to train a classical ML model (feature extraction using color histograms + HOG).
- `train_notebook.ipynb` - notebook scaffold showing dataset loading, feature extraction, training (RandomForest / SVM), and saving model to `model.pkl`.
- `disease_info.json` - sample mapping of class names to human-friendly descriptions and advice.
- `sample_data/` - placeholder for the PlantVillage dataset (not included).

## Notes
- The scaffold uses classical machine learning (RandomForest / SVM) rather than heavy CNN training. This is suitable for modest hardware.
- The app expects a trained `model.pkl` in the project root. If not present, the app will show instructions for training.

---
If you want, I can now:
- (A) Include a small dummy `model.pkl` trained on synthetic data so the app runs immediately for demonstration, or
- (B) Keep only scaffolding and let you train on full PlantVillage locally (recommended for real accuracy).

Tell me which option you prefer. 
