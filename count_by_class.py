from collections import Counter
import os
from glob import iglob

data_dir = r"sample_data\PlantVillage"
patterns = ['**/*.jpg','**/*.jpeg','**/*.png','**/*.JPG','**/*.JPEG','**/*.PNG']

files = []
for pat in patterns:
    for f in iglob(os.path.join(data_dir, pat), recursive=True):
        files.append(f)

print("Total image files found (recursive):", len(files))

# count by immediate parent folder
counts = Counter()
for f in files:
    parent = os.path.basename(os.path.dirname(f))
    counts[parent] += 1

print("Number of unique parent folders (labels):", len(counts))
# show first 12 classes with their counts
for cls, cnt in counts.most_common()[:12]:
    print(f"{cls:40s} : {cnt}")
