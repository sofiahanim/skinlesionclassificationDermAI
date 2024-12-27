import pandas as pd
from zipfile import ZipFile
from PIL import Image
import matplotlib.pyplot as plt
import os

#csv file
csv_path = '/mnt/data/test.csv'
data = pd.read_csv(csv_path)

print("First few rows of the CSV file:")
print(data.head())

if 'label' in data.columns:
    print("Unique labels in the CSV file:")
    print(data['label'].unique())
else:
    print("Label column not found. Available columns:")
    print(data.columns)


#zip file   

zip_path = '/mnt/data/test.zip'
extract_path = '/mnt/data/test_images'

with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

image_files = os.listdir(extract_path)
print("Extracted image files:", image_files)

if image_files:
    image_path = os.path.join(extract_path, image_files[0])
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title("Sample Image from ZIP")
    plt.show()
