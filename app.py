import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# --- הגדרות פרויקט ---
st.set_page_config(page_title="מערכת חכמה לחקלאות מדויקת", page_icon="🌾")

FILE_ID = '161ysydHCyvLOoVWkwWqJT5RpcMn_0rVu'
MODEL_PATH = 'best_resnet18_wheat.pt'

@st.cache_resource
def load_wheat_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('מוריד את המודל מהדרייב...'):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
    
    try:
        # טעינת הצ'קפוינט
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        
        # חילוץ שמות המחלקות מהקובץ עצמו!
        if isinstance(checkpoint, dict) and 'classes' in checkpoint:
            st.session_state['labels'] = checkpoint['classes']
        else:
            # ברירת מחדל אם לא נמצאו שמות
            st.session_state['labels'] = ["Brown Rust", "Healthy", "Leaf Rust", "Septoria", "Yellow Rust"]

        # בניית המודל עם מספר המחלקות הנכון (5)
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(st.session_state['labels']))
        
        # טעינת המשקולות
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model
    except Exception as e:
        st.error(f"שגיאה בטעינת המודל: {e}")
        return None

model = load_wheat_model()

# --- ממשק משתמש ---
st.title("זיהוי מחלות עלים בחיטה 🌾")
st.write("מבצעים: נבו הלר ומתן אדר | מנחה: אסי ברק")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.divider()
input_method = st.radio("בחר שיטת הזנה:", ("צילום במצלמה 📸", "העלאת תמונה מהגלריה 📁"))

img_file = st.camera_input("צלם") if input_method == "צילום במצלמה 📸" else st.file_uploader("בחר תמונה", type=['jpg','png','jpeg'])

if img_file is not None and model is not None:
    image = Image.open(img_file).convert('RGB')
    st.image(image, use_container_width=True)
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, prediction = torch.max(probabilities, 0)

    # שליפת השם הנכון מהלייבלים שגילינו בקובץ
    labels = st.session_state.get('labels', ["Unknown"] * 5)
    result_text = labels[prediction.item()]
    
    color = "green" if "Healthy" in result_text else "red"
    st.markdown(f"### אבחנה: :{color}[{result_text}]")
    st.write(f"**רמת ביטחון:** {confidence.item()*100:.2f}%")
