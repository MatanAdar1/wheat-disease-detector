import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown

# --- הגדרות פרויקט ועיצוב RTL ---
st.set_page_config(page_title="זיהוי מחלות חיטה 🌾", page_icon="🌾")

# הזרקת CSS ליישור לימין
st.markdown("""
    <style>
    .main {
        direction: rtl;
        text-align: right;
    }
    div[role="radiogroup"] {
        direction: rtl;
        text-align: right;
    }
    div.stMarkdown {
        text-align: right;
    }
    .stAlert {
        direction: rtl;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

FILE_ID = '161ysydHCyvLOoVWkwWqJT5RpcMn_0rVu'
MODEL_PATH = 'best_resnet18_wheat.pt'

# מילון תרגום והסברים
DISEASE_INFO = {
    "BlackPoint": {
        "heb": "חוד שחור (Black Point)",
        "desc": "שינוי צבע בקצה הגרעין או העלה, נגרם לרוב מעודף לחות.",
        "tip": "מומלץ לבחון את רמת הלחות בחלקה ולעקוב אחר התפשטות."
    },
    "FusariumFootRot": {
        "heb": "ריקבון בסיס הקנה (Fusarium)",
        "desc": "פטרייה התוקפת את בסיס הצמח וגורמת להצהבה וקמילה.",
        "tip": "חשוב למנוע השקיית יתר ולשקול טיפול פטרייתי ייעודי."
    },
    "HealthyLeaf": {
        "heb": "עלה בריא (Healthy)",
        "desc": "העלה נראה חיוני, ירוק וללא סימני מחלה פטרייתית.",
        "tip": "מצב מצוין! המשך בניטור קבוע של חלקת הניסוי."
    },
    "LeafBlight": {
        "heb": "קמלת עלים (Leaf Blight)",
        "desc": "כתמים מוארכים ויבשים על העלה המפחיתים את יכולת הפוטוסינתזה.",
        "tip": "יש לבדוק אם קיימת רגישות זנית ולמנוע הרטבת עלווה ישירה."
    },
    "WheatBlast": {
        "heb": "פיריקורליית החיטה (Wheat Blast)",
        "desc": "אחת המחלות הקשות ביותר, גורמת להלבנה מהירה של חלקים בצמח.",
        "tip": "זהירות! מחלה מידבקת מאוד. יש לבודד את הדגימה ולדווח למדריך."
    }
}

@st.cache_resource
def load_wheat_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('טוען מודל...'):
            gdown.download(f'https://drive.google.com/uc?id={FILE_ID}', MODEL_PATH, quiet=False)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        labels = checkpoint.get('classes', list(DISEASE_INFO.keys()))
        
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(labels))
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model.eval()
        return model, labels
    except Exception as e:
        st.error(f"שגיאה: {e}")
        return None, None

model, labels = load_wheat_model()

# --- ממשק משתמש ---
st.title("מערכת חכמה לזיהוי מחלות חיטה 🌾")
st.write("מבצעים: נבו הלר ומתן אדר | מנחה: אסי ברק")

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.divider()

input_method = st.radio("בחר כיצד להזין תמונה לבדיקה:", 
                        ("צילום במצלמה 📸", "העלאת תמונה מהגלריה 📁"))

if "מצלמה" in input_method:
    img_file = st.camera_input("צלם את העלה")
else:
    img_file = st.file_uploader("בחר קובץ תמונה (JPG, PNG, JPEG)", type=['jpg', 'png', 'jpeg'])

if img_file and model:
    image = Image.open(img_file).convert('RGB')
    st.image(image, caption="התמונה שנבחנה", use_container_width=True)
    
    with torch.no_grad():
        output = model(transform(image).unsqueeze(0))
        prob = torch.nn.functional.softmax(output[0], dim=0)
        conf, pred = torch.max(prob, 0)

    class_name = labels[pred.item()]
    info = DISEASE_INFO.get(class_name, {"heb": class_name, "desc": "", "tip": ""})

    st.divider()
    color = "green" if "Healthy" in class_name else "red"
    st.markdown(f"## אבחנה: :{color}[{info['heb']}]")
    
    # הצגת רמת ביטחון עם פס התקדמות ויזואלי
    st.write(f"**רמת ביטחון:** {conf.item()*100:.1f}%")
    st.progress(conf.item())
    
    with st.expander("מידע נוסף והמלצות לטיפול"):
        st.write(f"**תיאור המחלה:** {info['desc']}")
        st.info(f"**המלצה לניסוי:** {info['tip']}")
