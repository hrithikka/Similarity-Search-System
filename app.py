import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Loading the ResNet50 model
resnet = resnet.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V1)

#transformation applied to each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loading the features from file
f_dict = np.load("features.npy", allow_pickle=True).item()
f_list = list(f_dict.values())
image_names = list(f_dict.keys())

# Create a list of all image paths
image_paths = []
for root, dirs, files in os.walk("lfw_dir"):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            image_paths.append(os.path.join(root, file))

# Create a dictionary mapping image names to their corresponding paths
image_paths_dict = {}
for path in image_paths:
    name = os.path.basename(path)
    image_paths_dict[name] = path

# Create a NearestNeighbors model
model = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
model.fit(f_list)

# Define the Streamlit app
st.title('Similarity Search System')

# Create a sidebar with options for selecting an image or uploading an image
option = st.sidebar.selectbox('Select an option', ('Select an image', 'Upload an image'))

if option == 'Select an image':
    # Allow the user to select an image from the dataset
    image_name = st.selectbox('Select an image', image_names)

elif option == 'Upload an image':
    # Allow the user to upload an image
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Save the uploaded file to disk
        with open('query_image.jpg', 'wb') as f:
            f.write(uploaded_file.getvalue())
        # Load the uploaded image
        image_name = 'query_image.jpg'

# Display the query image
st.subheader('Query Image')
q_image = Image.open(image_paths_dict[image_name])
st.image(q_image, caption=image_name)

# Get the features of the query image
q_image = transform(q_image)
q_image = q_image.unsqueeze(0)
q_features = resnet(q_image).detach().numpy().flatten()

# Find the 10 most similar images to the query image
distances, indices = model.kneighbors([q_features])
st.subheader('Similar Images')
for i in indices[0]:
    similar_image = Image.open(image_paths[i])
    st.image(similar_image, caption=image_names[i])
