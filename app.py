import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import tempfile

# Loading the ResNet50 model
resnet = resnet.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V1)

#transformation applied to each of the image availabe n the data set
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loading the features from file where the features has been already obtained from colab
f_dict = np.load("features.npy", allow_pickle=True).item()
f_list = list(f_dict.values())
image_names = list(f_dict.keys())

# Creating a list of all image paths
image_paths = []
for root, dirs, files in os.walk("lfw_dir"):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            image_paths.append(os.path.join(root, file))

# Creating a dictionary for mapping image names to their corresponding paths
image_paths_dict = {}
for path in image_paths:
    name = os.path.basename(path)
    image_paths_dict[name] = path

# Creating a NearestNeighbors model
model = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
model.fit(f_list)

# Defining the Streamlit app
st.title('Similarity Search System')

# Creating a sidebar with options for selecting an image or uploading an image
option = st.sidebar.selectbox('Select an option', ('Select an image', 'Upload an image'))

if option == 'Select an image':
    #To select an image from the dataset
    image_name = st.selectbox('Select an image', image_names)
    # Displaying the query image
    st.subheader('Query Image')
    
    q_image_path = image_paths_dict.get(image_name)
    if q_image_path is not None:
        q_image = Image.open(q_image_path)
    else:
        st.write("Error: Could not find query image.")
        st.stop()

    st.image(q_image, caption=image_name)

elif option == 'Upload an image':
    #To upload an image
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
         # Saving the uploaded file to disk
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            image_name = temp_file.name
        # Displaying the query image
        st.subheader('Query Image')
        q_image = Image.open(image_name)
        st.image(q_image, caption=image_name)
    else:
       st.write("Please upload an image to continue.")



#Obtaing the features of the query image
q_image = transform(q_image)
q_image = q_image.unsqueeze(0)
q_features = resnet(q_image).detach().numpy().flatten()

# Find the 10 most similar images to the query image
distances, indices = model.kneighbors([q_features])
st.subheader('Similar Images')
for i in indices[0]:
    if i < len(image_paths):
        similar_image = Image.open(image_paths[i])
        st.image(similar_image, caption=image_names[i])
    else:
        st.write(f"Error: Could not find image with index {i}")

