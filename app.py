import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import requests

# --- הגדרות פרויקט ---
st.set_page_config(page_title="מערכת חכמה לחקלאות מדויקת", page_icon="🌾")

# --- הגדרות הורדה מדרייב ---
FILE_ID = '161ysydHCyvLOoVWkwWqJT5RpcMn_0rVu' 
MODEL_URL = 'https://docs.google.com/uc?export=download'
MODEL_PATH = 'best_resnet18_wheat.pt'

@st.cache_resource
def load_wheat_model():
    # 1. בדיקה והורדת המודל אם אינו קיים
    if not os.path.exists(MODEL_PATH):
        with st.spinner('מתחבר לדרייב וטוען את המודל המאומן... זה עשוי לקחת כדקה'):
            try:
                session = requests.Session()
                response = session.get(MODEL_URL, params={'id': FILE_ID}, stream=True)
                
                token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        token = value
                        break
                
                if token:
                    response = session.get(MODEL_URL, params={'id': FILE_ID, 'confirm': token}, stream=True)
                
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
            except Exception as e:
                st.error(f"שגיאה בתקשורת עם הדרייב: {e}")
                return None

    # 2. טעינת המודל לזיכרון
    try:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"שגיאה בטעינת המודל: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return None

# הפעלת טעינת המודל
model = load_wheat_model()

# --- ממשק משתמש ---
st.title("זיהוי מחלות עלים בחיטה 🌾")
st.markdown("### פרויקט מס' 3399 - אוניברסיטת תל אביב")
st.write("מבצעים: נבו הלר ומתן אדר | מנחה: אסי ברק")

# הגדרת עיבוד התמונה
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.divider()

# בחירת שיטת הזנה
input_method = st.radio("בחר כיצד להזין תמונה לבדיקה:", 
                        ("צילום במצלמה 📸", "העלאת תמונה מהגלריה 📁"))

if input_method == "צילום במצלמה 📸":
    img_file = st.camera_input("צלם את העלה")
else:
    img_file = st.file_uploader("בחר קובץ תמונה (JPG, PNG, JPEG)", type=['jpg', 'png', 'jpeg'])

# --- ביצוע הסיווג ---
if img_file is not None:
    if model is not None:
        image = Image.open(img_file).convert('RGB')
        st.image(image, caption="התמונה שנקלטה", use_container_width=True)
        
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, prediction = torch.max(probabilities, 0)

        st.divider()
        labels = ["בריא (Healthy)", "חולה (Diseased)"]
        color = "green" if prediction.item() == 0 else "red"
        
        st.markdown(f"### אבחנה: :{color}[{labels[prediction.item()]}]")
        st.write(f"**רמת ביטחון:** {confidence.item()*100:.2f}%")
        
        if prediction.item() == 1:
            st.warning("תיאור: המודל זיהה פגיעה בעלה. מומלץ לבדוק את תנאי הלחות וההשקיה.")
        else:
            st.success("תיאור: העלה נראה חיוני ותקין.")
    else:
        st.error("המודל לא נטען כראוי. נסה Reboot לאפליקציה.")
