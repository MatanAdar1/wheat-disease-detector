import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- Configuration & Page Setup ---
st.set_page_config(page_title="Wheat Disease Detector", page_icon="🌾", layout="centered")

# Define the classes (must match your training labels)
CLASS_NAMES = ['Black Point', 'Fusarium', 'Healthy', 'Leaf Blight', 'Wheat Blast']

# --- Model Loading ---
@st.cache_resource
def load_model():
    # Initialize ResNet18 architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    # Adjust final layer to the number of classes (5)
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    # Load your trained weights (ensure the file is in the same directory)
    model_path = "wheat_model.pth" 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model.eval()
    return model

model = load_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- UI Layout (English Only) ---
st.title("🌾 Wheat Guard AI")
st.subheader("Autonomous Disease Classification System")
st.write("Upload a leaf or grain image to receive an instant diagnostic report.")

uploaded_file = st.file_uploader("Choose an image sample...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Sample', use_container_width=True)
    
    st.write("---")
    with st.spinner('Analyzing sample using ResNet18...'):
        # Inference
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, index = torch.max(probabilities, 0)
        
        # Result Display
        label = CLASS_NAMES[index]
        conf_score = confidence.item() * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Diagnosis", label)
        with col2:
            st.metric("Confidence Score", f"{conf_score:.1f}%")

        # Visual feedback based on result
        if label == 'Healthy':
            st.success(f"The sample appears to be **Healthy**.")
        else:
            st.error(f"Potential **{label}** detected.")
            st.warning("**Recommendation:** Check local field humidity and consult with an agronomist if symptoms spread.")

# --- Sidebar Info ---
st.sidebar.title("System Info")
st.sidebar.info("""
**Project ID:** 3399  
**Model:** ResNet18 (Deep Learning)  
**Hardware:** ESP32 Integration Ready  
""")
