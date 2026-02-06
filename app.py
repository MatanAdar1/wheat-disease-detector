import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- הגדרות דף ---
st.set_page_config(page_title="מערכת חכמה לחקלאות מדויקת", page_icon="🌾")
st.title("זיהוי מחלות עלים בחיטה 🌾")
st.markdown("### פרויקט מס' 3399 - אוניברסיטת תל אביב")
st.write("מבצעים: נבו הלר ומתן אדר | מנחה: אסי ברק")

# --- פונקציה לטעינת המודל ---
@st.cache_resource
def load_wheat_model():
    # בניית ארכיטקטורת ResNet18
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    # הנחה: 2 מחלקות (בריא/חולה) לפי התוכנית
    model.fc = nn.Linear(num_ftrs, 2) 
    
    # טעינת המשקולות (וודאו שהקובץ נמצא באותה תיקייה)
    model_path = 'best_resnet18_wheat.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        st.error(f"קובץ המודל {model_path} לא נמצא בתיקייה!")
        return None

model = load_wheat_model()

# --- הכנת התמונה (Preprocessing) ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- ממשק משתמש ---
st.divider()
option = st.radio("כיצד תרצה להזין תמונה?", ("צלם במצלמה", "העלה קובץ מהגלריה"))

if option == "צלם במצלמה":
    img_file = st.camera_input("צלם את עלה החיטה")
else:
    img_file = st.file_uploader("בחר תמונה", type=['jpg', 'png', 'jpeg'])

# --- ביצוע הסיווג ---
if img_file is not None and model is not None:
    image = Image.open(img_file).convert('RGB')
    st.image(image, caption="התמונה שנקלטה", use_container_width=True)
    
    # עיבוד
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, prediction = torch.max(probabilities, 0)

    # --- הצגת תוצאות ---
    st.divider()
    labels = ["בריא (Healthy)", "חולה (Diseased)"]
    result = labels[prediction.item()]
    prob_val = confidence.item() * 100

    if prediction.item() == 0:  # בריא
        st.success(f"### אבחנה: {result}")
        st.write(f"**רמת ביטחון:** {prob_val:.2f}%")
        st.info("תיאור: העלה נראה חיוני ותקין. המשך מעקב אחר מדדי השקיה.")
    else:  # חולה
        st.error(f"### אבחנה: {result}")
        st.write(f"**רמת ביטחון:** {prob_val:.2f}%")
        st.warning("תיאור: זוהו פתוגנים או סימני מחלה על העלה. מומלץ לבדוק את קבוצת הניסוי בהשוואה לבקרה.")

st.sidebar.markdown("---")
st.sidebar.write("מערכת זו מבוססת על מודל ResNet18 שאומן על דאטה-סט של מחלות חיטה, כחלק מפרויקט גמר בהנדסת חשמל.")