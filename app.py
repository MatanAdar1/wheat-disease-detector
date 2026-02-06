import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# --- הגדרות פרויקט ---
st.set_page_config(page_title="מערכת חכמה לחקלאות מדויקת", page_icon="🌾")

# --- הגדרות הורדה מדרייב ---
FILE_ID = '161ysydHCyvLOoVWkwWqJT5RpcMn_0rVu'
MODEL_PATH = 'best_resnet18_wheat.pt'

@st.cache_resource
def load_wheat_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('מוריד את המודל מהדרייב...'):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
    
    try:
        # 1. בניית הארכיטקטורה
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        # 2. טעינת ה"חבילה" (Checkpoint)
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        
        # 3. שליפת המודל מתוך ה-Dictionary (לפי השגיאה שקיבלת Key: 'model_state_dict')
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

if input_method == "צילום במצלמה 📸":
    img_file = st.camera_input("צלם את העלה")
else:
    img_file = st.file_uploader("בחר קובץ תמונה", type=['jpg','png','jpeg'])

if img_file is not None and model is not None:
    image = Image.open(img_file).convert('RGB')
    st.image(image, use_container_width=True)
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, prediction = torch.max(probabilities, 0)

    res = ["בריא (Healthy)", "חולה (Diseased)"]
    color = "green" if prediction.item() == 0 else "red"
    st.markdown(f"### אבחנה: :{color}[{res[prediction.item()]}]")
    st.write(f"**רמת ביטחון:** {confidence.item()*100:.2f}%")
