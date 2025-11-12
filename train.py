# train.py
import os
from glob import iglob
from collections import defaultdict
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from model_utils import preprocess_image_for_ml
import random
import sys

# Path to dataset - keep as absolute or relative to project root
DATA_DIR = r"C:\Users\bhuvan\Downloads\PlantDiseaseDetector_Project\sample_data\PlantVillage"

# Set this lower while experimenting to speed up runs
LIMIT_PER_CLASS = 300   # set None to use all images
RANDOM_SEED = 42

def load_data(data_dir, limit_per_class=None):
    """
    Recursively find image files under data_dir, use immediate parent folder name as label.
    Returns X (np.ndarray shape (N,features)) and y (np.ndarray shape (N,))
    """
    patterns = ['**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.JPG', '**/*.JPEG', '**/*.PNG']
    files_by_class = defaultdict(list)

    # gather files recursively
    for pat in patterns:
        search = os.path.join(data_dir, pat)
        for f in iglob(search, recursive=True):
            # skip directories
            if not os.path.isfile(f):
                continue
            parent = os.path.basename(os.path.dirname(f))
            files_by_class[parent].append(f)

    classes = sorted(files_by_class.keys())
    print("Classes found (parent-folder labels):", len(classes))
    if len(classes) == 0:
        print("No image files found under:", data_dir)
        return np.array([]), np.array([])

    # show a quick summary (first few classes and counts)
    sample_summary = {c: len(files_by_class[c]) for c in classes[:12]}
    print("Sample class counts (first 12):", sample_summary)

    X = []
    y = []

    for cls in classes:
        files = files_by_class[cls]
        if limit_per_class is not None:
            files = files[:limit_per_class]
        # small progress print per class
        print(f"Processing class '{cls}' ({len(files)} files)...", end='', flush=True)
        counted = 0
        for fp in files:
            try:
                img = Image.open(fp).convert('RGB')
                feat = preprocess_image_for_ml(img)   # uses your existing function
                X.append(feat)
                y.append(cls)
                counted += 1
            except Exception as e:
                # skip unreadable files, optionally log them
                # print(" Skip unreadable:", fp, e)
                continue
        print(f" done ({counted} added)")

    if len(X) == 0:
        return np.array([]), np.array([])

    # stack feature vectors (handle case where each feat is 1-D)
    try:
        X_arr = np.vstack(X)
    except Exception as e:
        # fallback: convert to numpy array (may produce object dtype if inconsistent lengths)
        X_arr = np.array(X)
    y_arr = np.array(y)
    return X_arr, y_arr

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Loading data from:", DATA_DIR)
    X, y = load_data(DATA_DIR, LIMIT_PER_CLASS)
    print('X shape', X.shape, 'y shape', y.shape)

    # Basic sanity checks
    if X.size == 0 or y.size == 0:
        print("\nERROR: No training data found. Please check DATA_DIR and folder structure.")
        print("Expected structure: sample_data/PlantVillage/<class_name>/*.jpg (or nested).")
        sys.exit(1)

    unique_labels = np.unique(y)
    print("Unique classes discovered:", len(unique_labels))
    if len(unique_labels) < 2:
        print("\nERROR: Need at least 2 classes to train. Found only:", unique_labels)
        sys.exit(1)

    # Split data (stratify by label)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        "SVM": SVC(kernel='rbf', probability=True)
    }

    best_acc = 0.0
    best_model = None
    best_name = None

    for name, clf in models.items():
        print(f"\nTraining {name} ...")
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Training failed for {name}: {e}")
            continue

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

        if acc > best_acc:
            best_acc = acc
            best_model = clf
            best_name = name

    if best_model is not None:
        joblib.dump(best_model, 'model.pkl')
        print(f"\nSaved best model: {best_name} with acc={best_acc:.4f} -> model.pkl")
    else:
        print("\nNo model trained successfully.")
