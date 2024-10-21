import os
import json
from sklearn.model_selection import train_test_split

# DATA_DIR = '/data/ephemeral/home/dataset/'
DATA_DIR = '/data/ephemeral/home/github/proj2/'
file_name= 'annotation'
with open(os.path.join(DATA_DIR, file_name+'.json')) as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']

# Create a dictionary to map image IDs to their corresponding annotations
image_annotations = {}
for annotation in annotations:
    annotation['segmentation'] = [[0, 0, 0, 0, 0, 0, 0, 0]]
    image_annotations.setdefault(annotation['image_id'], []).append(annotation)

# Create the JSON objects
image_data = {
    'info': data['info'],
    'licenses': data['licenses'],
    'images': images,
    'annotations': annotations,
    'categories': data['categories']
}


# Remove existing files
if os.path.exists(os.path.join(DATA_DIR, file_name+'fseg.json')):
    os.remove(os.path.join(DATA_DIR, file_name+'fseg.json'))
    print("Removed the existing false segmentation json")
# Save the training and validation JSON objects to file
with open(os.path.join(DATA_DIR, file_name+'fseg.json'), 'w') as f:
    json.dump(image_data, f)
    print("Successfully created the {file_name}_fseg.json!")
