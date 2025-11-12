import os
d = r'sample_data\PlantVillage'
d2 = r'sample_data\archive (4)'
print("CWD:", os.getcwd())
print("Exists PlantVillage ->", os.path.exists(d))
print("Exists archive (4) ->", os.path.exists(d2))
if os.path.exists(d):
    print("sample_data\\PlantVillage first 5 classes:", os.listdir(d)[:5])
elif os.path.exists(d2):
    print("sample_data\\archive (4) first 5 classes:", os.listdir(d2)[:5])
else:
    print("Neither folder exists. sample_data contents:", os.listdir('sample_data'))
