import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="ניהול ומאגר ניסוי חיטה 🌾", page_icon="🌾", layout="wide")

# עיצוב ממוקד למניעת קריסת הטקסט האנכית (מטפל בבעיה מ-image_5a81bc.png)
st.markdown("""
    <style>
    /* יישור טקסט וכיווניות רק לאלמנטים של תוכן, ללא שבירת השלד של האתר */
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, label, [data-testid="stWidgetLabel"] {
        text-align: right !important;
        direction: rtl !important;
    }
    /* יישור כפתורים ותיבות טקסט לימין */
    .stButton>button, .stSelectbox, .stTextArea {
        direction: rtl !important;
        text-align: right !important;
    }
    /* יישור מדדים (Metrics) */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        text-align: right !important;
        direction: rtl !important;
    }
    /* התאמת הטבלה האופקית לחלוטין לימין */
    [data-testid="stDataFrame"] {
        direction: rtl !important;
        text-align: right !important;
    }
    /* עיצוב כרטיסיות מידע (Cards) משודרגות ואטרקטיביות */
    .custom-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-right: 5px solid #4caf50;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

FILE_ID = '161ysydHCyvLOoVWkwWqJT5RpcMn_0rVu'
MODEL_PATH = 'best_resnet18_wheat.pt'
CONFIDENCE_THRESHOLD = 0.25

# אתחול היסטוריית הצמחים בזיכרון המערכת
if "plant_history" not in st.session_state:
    st.session_state.plant_history = {}

# אתחול אינדקס הצמח הנוכחי לצורך דפדוף
if "current_plant_idx" not in st.session_state:
    st.session_state.current_plant_idx = 0

DISEASE_INFO = {
    "BlackPoint": {"heb": "חוד שחור (Black Point)"},
    "FusariumFootRot": {"heb": "ריקבון בסיס הקנה (Fusarium)"},
    "HealthyLeaf": {"heb": "עלה בריא (Healthy)"},
    "LeafBlight": {"heb": "קמלת עלים (Leaf Blight)"},
    "WheatBlast": {"heb": "פיריקורליית החיטה (Wheat Blast)"}
}

@st.cache_data
def load_experiment_data():
    if not os.path.exists("plants_experiment_73.csv"):
        return None
    return pd.read_csv("plants_experiment_73.csv")

@st.cache_resource
def load_wheat_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('טוען מודל זיהוי...'):
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
        return None, None

plants_df = load_experiment_data()

if plants_df is None:
    st.title("🌾 מערכת ניסוי חיטה חכמה")
    st.error("❌ קובץ הנתונים plants_experiment_73.csv חסר בשרת!")
    st.stop()

model, labels = load_wheat_model()

st.title("🌾 מערכת חכמה לניהול, ניטור וזיהוי מחלות חיטה")
st.write("מבצעים: נבו הלר ומתן אדר | מנחה: אסי ברק")
st.divider()

# יצירת רשימת צמחים ייחודית מסודרת
plants_df['select_label'] = plants_df.apply(lambda r: f"ID: {r['id']} | שם: {r['name']}", axis=1)
unique_plants = list(plants_df['select_label'].unique())

st.subheader("🕹️ בקרת ניווט ודפדוף בין צמחי הניסוי")
nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

with nav_col1:
    if st.button("➡️ הצמח הקודם", use_container_width=True):
        if st.session_state.current_plant_idx > 0:
            st.session_state.current_plant_idx -= 1
            st.rerun()

with nav_col3:
    if st.button("הצמח הבא ⬅️", use_container_width=True):
        if st.session_state.current_plant_idx < len(unique_plants) - 1:
            st.session_state.current_plant_idx += 1
            st.rerun()

with nav_col2:
    selected_label = st.selectbox(
        "או בחר ישירות מהרשימה:", 
        unique_plants, 
        index=st.session_state.current_plant_idx,
        key="plant_selector"
    )
    # עדכון האינדקס במידה והמשתמש בחר ישירות מהתיבה
    st.session_state.current_plant_idx = unique_plants.index(selected_label)

# שליפת נתוני הצמח הנבחר
plant_row = plants_df[plants_df['select_label'] == selected_label].iloc[0]
plant_id = int(plant_row['id'])
plant_name = str(plant_row['name'])

st.markdown("<br>", unsafe_allow_html=True)

# כרטיסיית סיכום עליונה מעוצבת
with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔢 מזהה ייחודי (ID)", str(plant_id))
    col2.metric("🌱 שם הצמח", plant_name)
    col3.metric("🧪 סוג טיפול", str(plant_row['#Treatment']))
    col4.metric("📉 מדד עקה (Stress)", f"{plant_row['stressDegree']:.3f}")

st.markdown("<br>", unsafe_allow_html=True)

# אזור הערכה כללית משולב עם עדכונים מהשטח בזמן אמת
st.subheader("📋 הערכה כללית ומצב צמח עדכני")
treatment = plant_row['#Treatment']
stress = plant_row['stressDegree']

status_html = ""
if treatment == 'Drought' and stress > 0.15:
    status_html = f"<div class='custom-card' style='border-right-color: #f44336;'>⚠️ <b>סטטוס פנוטיפי:</b> עקת יובש משמעותית (מדד: {stress:.3f}). הצמח מציג סימני מחסור חריפים במים.</div>"
elif treatment == 'Drought':
    status_html = f"<div class='custom-card' style='border-right-color: #ff9800;'>🔸 <b>סטטוס פנוטיפי:</b> עקת יובש מתונה (מדד: {stress:.3f}). הצמח נמצא תחת מגבלת השקיה מבוקרת.</div>"
else:
    status_html = f"<div class='custom-card'>✅ <b>סטטוס פנוטיפי:</b> תקין ויציב. קבוצת ביקורת (Control), משטר השקיה מלא.</div>"

st.markdown(status_html, unsafe_allow_html=True)

# כאן הטקסט והאבחון מוזרקים דינמית לתוך המידע הכללי ברגע שמעלים תמונה
if plant_id in st.session_state.plant_history and len(st.session_state.plant_history[plant_id]) > 0:
    latest_record = st.session_state.plant_history[plant_id][-1]
    st.markdown(f"""
    <div class="custom-card" style="border-right-color: #2196f3; background-color: #e3f2fd;">
        🔍 <b>עדכון פתולוגי משולב מהשטח ({latest_record['timestamp']}):</b><br><br>
        <b>מצב פתולוגי שזוהה:</b> {latest_record['diagnosis']}<br>
        <b>תיאור מצב הצמח העדכני:</b> {latest_record['notes']}
    </div>
    """, unsafe_allow_html=True)

st.divider()

# טבלה אופקית (כמו באקסל - תואם ל-image_5a8926.png)
st.subheader("📊 נתוני הצמח המלאים מתוך הניסוי")
all_cols = list(plants_df.columns)
if 'id' in all_cols: all_cols.remove('id')
if 'name' in all_cols: all_cols.remove('name')
if 'select_label' in all_cols: all_cols.remove('select_label')
ordered_cols = ['id', 'name'] + all_cols

single_plant_df = plants_df[plants_df['id'] == plant_id][ordered_cols].copy()
st.dataframe(single_plant_df, use_container_width=True, hide_index=True)

st.divider()

# ממשק צילום והוספה למאגר
st.subheader("📸 בדיקה חזותית והוספה למאגר התיעודים")
with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        input_method = st.radio("בחר דרך להזנת תמונה:", 
                                ("צילום ישיר במצלמה 📸", "העלאת קובץ מהגלריה 📁"), key="input_meth")
        if "מצלמה" in input_method:
            img_file = st.camera_input("צלם את העלה", key="capture_photo")
        else:
            img_file = st.file_uploader("בחר קובץ תמונה", type=['jpg', 'png', 'jpeg'], key="upload_photo")
            
    with c2:
        user_notes = st.text_area("✍️ תיאור ומצב הצמח בזמן הצילום:", 
                                  placeholder="הקלד כאן תיאור מילולי של סימפטומים, גודל, צבע או הערות מיוחדות מהחממה...", height=150)

    if img_file:
        image = Image.open(img_file).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        auto_diagnosis = "לא הופעל אבחון"
        if model:
            with torch.no_grad():
                output = model(transform(image).unsqueeze(0))
                prob = torch.nn.functional.softmax(output[0], dim=0)
                conf, pred = torch.max(prob, 0)
            
            if conf.item() < CONFIDENCE_THRESHOLD:
                auto_diagnosis = "לא זוהה עלה רלוונטי"
            else:
                class_name = labels[pred.item()]
                auto_diagnosis = DISEASE_INFO.get(class_name, {"heb": class_name})["heb"]
        
        st.markdown(f"**🔍 תוצאת ניתוח חזותי:** {auto_diagnosis}")
        
        if st.button(f"💾 שמור צילום ותיאור מצב למאגר של צמח {plant_name}", use_container_width=True):
            current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if plant_id not in st.session_state.plant_history:
                st.session_state.plant_history[plant_id] = []
                
            st.session_state.plant_history[plant_id].append({
                "timestamp": current_time,
                "image": image,
                "notes": user_notes if user_notes else "לא הוכנס פירוט חופשי",
                "diagnosis": auto_diagnosis
            })
            st.success("הנתונים נשמרו בהצלחה וצורפו ישירות להערכה הכללית של הצמח למעלה!")
            st.rerun()

st.divider()

# היסטוריית תיעודים מפורטת בתחתית הדף
st.subheader(f"🗄️ היסטוריית צילומים ותיעודים עבור צמח {plant_name}")
if plant_id in st.session_state.plant_history and len(st.session_state.plant_history[plant_id]) > 0:
    for record in reversed(st.session_state.plant_history[plant_id]):
        with st.container(border=True):
            hc1, hc2 = st.columns([1, 3])
            with hc1:
                st.image(record["image"], use_container_width=True)
            with hc2:
                st.markdown(f"### 📅 תאריך ושעה: `{record['timestamp']}`")
                st.markdown(f"**🔬 אבחון פתולוגי:** {record['diagnosis']}")
                st.markdown(f"**📝 תיאור מצב שהוזן:** {record['notes']}")
else:
    st.info("אין עדיין צילומים מתועדים במאגר עבור צמח זה.")
