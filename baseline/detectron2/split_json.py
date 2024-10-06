import os
import json
from sklearn.model_selection import train_test_split

DATA_DIR = '/data/ephemeral/home/dataset/'
with open(os.path.join(DATA_DIR, 'train.json')) as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']

# Create a dictionary to map image IDs to their corresponding annotations
image_annotations = {}
for annotation in annotations:
    image_annotations.setdefault(annotation['image_id'], []).append(annotation)

# Split the images and their corresponding annotations
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

train_annotations = []
for image in train_images:
    train_annotations.extend(image_annotations[image['id']])
val_annotations = []
for image in val_images:
    val_annotations.extend(image_annotations[image['id']])

# Create the training and validation JSON objects
train_data = {
    'info': data['info'],
    'licenses': data['licenses'],
    'images': train_images,
    'annotations': train_annotations,
    'categories': data['categories']
}
val_data = {
    'info': data['info'],
    'licenses': data['licenses'],
    'images': val_images,
    'annotations': val_annotations,
    'categories': data['categories']
}

# Remove existing files
if os.path.exists(os.path.join(DATA_DIR, 'train2.json')):
    os.remove(os.path.join(DATA_DIR, 'train2.json'))
    print("Removed the existing train2 json")
if os.path.exists(os.path.join(DATA_DIR, 'val2.json')):
    os.remove(os.path.join(DATA_DIR, 'val2.json'))
    print("Removed the existing val2 json")

# Save the training and validation JSON objects to file
with open(os.path.join(DATA_DIR, 'train2.json'), 'w') as f:
    json.dump(train_data, f)
    print("Successfully created the train2.json!")
with open(os.path.join(DATA_DIR, 'val2.json'), 'w') as f:
    json.dump(val_data, f)
    print("Successfully created the val2.json!")
