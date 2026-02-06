import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import requests

# --- הגדרות פרויקט ---
st.set_page_config(page_title="מערכת חכמה לחקלאות מדויקת", page_icon="🌾")

# --- הגדרות הורדה מדרייב (ה-ID מהלינק שלך מוטמע כאן) ---
FILE_ID = '161ysydHCyvLOoVWkwWqJT5RpcMn_0rVu' 
MODEL_URL = f'https://docs.google.com/uc?export=download&id={FILE_ID}'
MODEL_PATH = 'best_resnet18_wheat.pt'

@st.cache_resource
def load_wheat_model():
    # הורדת המודל מהדרייב רק אם הוא לא קיים בשרת האפליקציה
    if not os.path.exists(MODEL_PATH):
        with st.spinner('מתחבר לדרייב וטוען את המודל המאומן... זה עשוי לקחת כדקה'):
            try:
                response = requests.get(MODEL_URL, stream=True)
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                st.error(f"שגיאה בהורדת המודל מהדרייב: {e}")
                return None

    # בניית הארכיטקטורה (ResNet18) וטעינת המשקולות
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # 2 מחלקות: בריא/חולה
    
    # טעינה למעבד (CPU) - מתאים לשרתים חינמיים
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_wheat_model()

# --- ממשק משתמש ---
st.title("זיהוי מחלות עלים בחיטה 🌾")
st.markdown("### פרויקט מס' 3399 - אוניברסיטת תל אביב")
st.write("מבצעים: נבו הלר ומתן אדר | מנחה: אסי ברק")

# הגדרת עיבוד התמונה (חייב להיות זהה למה שהשותף עשה באימון)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.divider()
img_file = st.camera_input("צלם את עלה החיטה לבדיקה")

if img_file is not None and model is not None:
    # הצגת התמונה ועיבוד
    image = Image.open(img_file).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, prediction = torch.max(probabilities, 0)

    # הצגת תוצאות
    st.divider()
    labels = ["בריא (Healthy)", "חולה (Diseased)"]
    color = "green" if prediction.item() == 0 else "red"
    
    st.markdown(f"### אבחנה: :{color}[{labels[prediction.item()]}]")
    st.write(f"**רמת ביטחון:** {confidence.item()*100:.2f}%")
    
    if prediction.item() == 1:
        st.warning("תיאור: זוהו סימני פגיעה בעלה. מומלץ לבדוק את תנאי ההשקיה והלחות.")
    else:
        st.success("תיאור: העלה נראה בריא ותקין.")
