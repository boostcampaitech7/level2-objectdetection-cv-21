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
    annotation['segmentation'] = [[0, 0, 0, 0, 0, 0, 0, 0]]
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
if os.path.exists(os.path.join(DATA_DIR, 'train_split.json')):
    os.remove(os.path.join(DATA_DIR, 'train_split.json'))
    print("Removed the existing train_split json")
if os.path.exists(os.path.join(DATA_DIR, 'val_split.json')):
    os.remove(os.path.join(DATA_DIR, 'val_split.json'))
    print("Removed the existing val_split json")

# Save the training and validation JSON objects to file
with open(os.path.join(DATA_DIR, 'train_split.json'), 'w') as f:
    json.dump(train_data, f)
    print(f"Successfully created the train_split.json! The total size is {len(train_images)}")
with open(os.path.join(DATA_DIR, 'val_split.json'), 'w') as f:
    json.dump(val_data, f)
    print(f"Successfully created the val_split.json! The total size is {len(val_images)}")
