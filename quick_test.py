# quick_test.py — automatic random test
import os, glob, random, joblib, numpy as np
from PIL import Image
from model_utils import preprocess_image_for_ml

# 1️⃣ recursively collect all JPG/JPEG/PNG files
base_dir = os.path.join("sample_data", "PlantVillage")
files = glob.glob(os.path.join(base_dir, "**", "*.*"), recursive=True)
files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if not files:
    raise SystemExit(f"No image files found under {base_dir}. Please check your dataset path.")

# 2️⃣ choose one random image
img_path = random.choice(files)
print(f"Randomly selected image:\n  {img_path}")

# 3️⃣ load the model
model_path = "model.pkl"
if not os.path.exists(model_path):
    raise SystemExit("model.pkl not found! Train first using train.py.")
model = joblib.load(model_path)

# 4️⃣ preprocess the image
img = Image.open(img_path).convert("RGB")
X = preprocess_image_for_ml(img)

# 5️⃣ predict and display top-3 results
pred = model.predict([X])[0]
print(f"\nPredicted class: {pred}")

if hasattr(model, "predict_proba"):
    probs = model.predict_proba([X])[0]
    classes = list(model.classes_)
    top_idx = np.argsort(probs)[::-1][:3]
    print("\nTop-3 predictions:")
    for i in top_idx:
        print(f"  {classes[i]:60s} → {probs[i]:.3f}")
else:
    print("Model does not support probability estimates.")
