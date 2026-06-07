import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown
import pandas as pd
import json
import glob
from datetime import datetime

st.set_page_config(page_title="מערכת לזיהוי מחלות צמחים 🌾", page_icon="🌾", layout="wide")

# ניהול ניווט העמודים באפליקציה
if "page" not in st.session_state:
    st.session_state.page = "home"

# עיצוב ממוקד ומניעת קריסת טקסטים אנכית (RTL)
st.markdown("""
    <style>
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, label, [data-testid="stWidgetLabel"] {
        text-align: right !important;
        direction: rtl !important;
    }
    .stButton>button, .stSelectbox, .stTextArea {
        direction: rtl !important;
        text-align: right !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        text-align: right !important;
        direction: rtl !important;
    }
    [data-testid="stDataFrame"] {
        direction: rtl !important;
        text-align: right !important;
    }
    .custom-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-right: 5px solid #4caf50;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .home-box {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    </style>
    """, unsafe_allow_html=True)

FILE_ID = '161ysydHCyvLOoVWkwWqJT5RpcMn_0rVu'
MODEL_PATH = 'best_resnet18_wheat.pt'
CONFIDENCE_THRESHOLD = 0.25
HISTORY_DIR = "saved_plant_history"

os.makedirs(HISTORY_DIR, exist_ok=True)

if "current_plant_idx" not in st.session_state:
    st.session_state.current_plant_idx = 0

DISEASE_INFO = {
    "BlackPoint": {
        "heb": "חוד שחור (Black Point)",
        "desc": "מחלה הנגרמת על ידי קומפלקס פטריות או תנאים סביבתיים בשלבי הבשלת הגרעין. היא מתאפיינת בהשחרה או התכהות של קצה הגרעין או אזורים בעלה, ומתפתחת בעיקר בעקבות לחות גבוהה מאוד, גשמים ממושכים או טל כבד בתקופת מילוי הגרגר.",
        "tip": "מומלץ להפחית את משטר ההשקיה בשלבי ההבשלה ולמנוע לחות עודפת בשדה. יש להקפיד על אוורור נאות של השטח, שימוש בזרעים נקיים ומחוטאים בעונה הבאה, ובמקרה של נגיעות מוקדמת ונרחבת לשקול שילוב קוטלי פטריות מותאמים."
    },
    "FusariumFootRot": {
        "heb": "ריקבון בסיס הקנה (Fusarium)",
        "desc": "מחלה פטרייתית קרקעית עקשנית התוקפת את מערכת השורשים ובסיס הקנה. הפטרייה חוסמת את צינורות ההובלה של הצמח, מובילה להצהבה, נבילה, וריקבון חום בבסיס הגבעול, ובסופו של דבר גורמת לקמילת הצמח ולהיווצרות שיבולים לבנות וריקות מגרגרים.",
        "tip": "יש ליישם מחזור זרעים קפדני עם גידולים שאינם דגניים (כגון קטניות) למשך שנתיים לפחות להפחתת עומס הפתוגן בקרקע. בנוסף, יש להימנע מהשקיית יתר, לדאוג לניקוז יעיל של השדה, להשתמש בזנים עמידים ובמידת הצורך לבצע חיטוי זרעים ייעודי לפני הזריעה הבאה."
    },
    "HealthyLeaf": {
        "heb": "עלה בריא (Healthy)",
        "desc": "העלה מציג חיוניות גבוהה, צבע ירוק אחיד ושטח פנים נקי לחלוטין מכתמים, גלדים או סימני תקיפה פטרייתית כלשהי. תהליך הפוטוסינתזה מתנהל בצורה אופטימלית המאפשרת התפתחות תקינה ומלאה של הצמח ויבול פוטנציאלי גבוה.",
        "tip": "מצב מצוין! מומלץ להמשיך במשטר הטיפוח וההשקיה המאוזן הנוכחי, להקפיד על דישון מותאם (חנקן, זרחן ואשלגן) לפי שלב הגידול הקיים, ולשמור על ניטור שבועי קבוע בכל חלקי החלקה כדי לזהות סימנים ראשוניים של פגעים מבעוד מועד."
    },
    "LeafBlight": {
        "heb": "קמלת עלים (Leaf Blight)",
        "desc": "מחלה פטרייתית המתבטאת בהופעת כתמים מוארכים, יבשים וחומים-אפרפרים על גבי העלים. הכתמים מתרחבים ומתחברים זה לזה, מביאים להתייבשות נרחבת של רקמת העלה, מפחיתים דרסטית את כושר הפוטוסינתזה של הצמח ופוגעים במשקל הגרעין וביבול הסופי.",
        "tip": "מומלץ לשלב ריסוס בקוטלי פטריות רחבי טווח עם זיהוי הסימנים הראשוניים על עלי הבסיס כדי למנוע את עליית המחלה לעלה הדגל. למניעה עתידית, יש להשתמש בזרעים נקיים ומאושרים, ליישם מחזור זרעים, להשמיד שאריות צמחים נגועות מהעונה הקודמת ולהימנע ככל הניתן מהשקיה בהמטרה שמסייעת להפצת הנבגים על העלווה."
    },
    "WheatBlast": {
        "heb": "פיריקורליית החיטה (Wheat Blast)",
        "desc": "מחלה פטרייתית הרסנית ואגרסיבית ביותר המסוגלת לתקוף את כל חלקי הצמח. הפגיעה הקשה ביותר מתרחשת בשיבולת, שהופכת ללבנה ויבשה לחלוטין תוך ימים ספורים, מה שמונע לחלוטין את התפתחות הגרגרים ומוביל לאובדן יבול מוחלט בשיבולים הנגועות.",
        "tip": "זהו מצב חירום חקלאי בשל מהירות ההתפשטות הסחרחרה של המחלה. יש לבודד מיד את האזור הנגוע, להפסיק לחלוטין השקיה עילית (המטרה) כדי למנוע את הפצת הנבגים ברוח, ולרסס בדחיפות בקוטלי פטריות סיסטמיים חזקים (כגון טריאזולים או סטרובילורינים) בהתאם להנחיות העדכניות של הגנת הצומח."
    }
}

def save_record_to_disk(plant_id, image, diagnosis, notes, class_name):
    plant_folder = os.path.join(HISTORY_DIR, str(plant_id))
    os.makedirs(plant_folder, exist_ok=True)
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    img_path = os.path.join(plant_folder, f"{timestamp_file}.png")
    image.save(img_path)
    
    meta_path = os.path.join(plant_folder, f"{timestamp_file}.json")
    meta_data = {"timestamp": display_time, "diagnosis": diagnosis, "class_name": class_name, "notes": notes}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)

def load_records_from_disk(plant_id):
    records = []
    plant_folder = os.path.join(HISTORY_DIR, str(plant_id))
    if not os.path.exists(plant_folder): return records
    json_files = glob.glob(os.path.join(plant_folder, "*.json"))
    json_files.sort()
    for json_path in json_files:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        img_path = os.path.join(plant_folder, f"{base_name}.png")
        if os.path.exists(img_path):
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["image_path"] = img_path
            records.append(meta)
    return records

@st.cache_data
def load_experiment_data():
    if not os.path.exists("plants_experiment_73.csv"): return None
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
    except:
        return None, None

plants_df = load_experiment_data()
model, labels = load_wheat_model()

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 🏠 דף הבית של האפליקציה
# ==========================================
if st.session_state.page == "home":
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🌾 מערכת לזיהוי מחלות צמחים</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #555;'>מבצעים: נבו הלר ומתן אדר | מנחה: אסי ברק</p>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="home-box">
            <h3 style='color: #1565c0;'>📸 אבחון חזותי מהיר</h3>
            <p>צילום או העלאת עלה לזיהוי מיידי של סוג המחלה, קבלת פירוט מורחב והמלצות טיפול מעשיות (חד-פעמי ללא טבלה).</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("פתח אבחון מהיר 🚀", use_container_width=True, key="btn_single"):
            st.session_state.page = "single_diagnosis"
            st.rerun()
            
    with col2:
        st.markdown("""
        <div class="home-box">
            <h3 style='color: #2e7d32;'>📊 ניהול ומעקב ניסוי חיטה</h3>
            <p>צפייה בטבלת הנתונים המלאה של הניסוי, ניווט ודפדוף בין צמחים, והוספת צילומים קבועים למאגר ההיסטורי של כל צמח.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("פתח מערכת ניסוי ומאגר 🔬", use_container_width=True, key="btn_exp"):
            st.session_state.page = "experiment_management"
            st.rerun()

# ==========================================
# 📸 עמוד 1: אבחון חד-פעמי (מהיר)
# ==========================================
elif st.session_state.page == "single_diagnosis":
    if st.button("🔙 חזרה לדף הבית", key="back_home_1"):
        st.session_state.page = "home"
        st.rerun()
        
    st.header("📸 אבחון חזותי מהיר והמלצות טיפול")
    st.write("ממשק צילום עצמאי לבדיקת פגעים ומחלות בעלים בזמן אמת")
    st.divider()
    
    c1, c2 = st.columns([1, 1])
    with c1:
        input_method = st.radio("בחר דרך להזנת תמונה:", ("צילום ישיר במצלמה 📸", "העלאת קובץ מהגלריה 📁"), key="single_meth")
        if "מצלמה" in input_method:
            img_file = st.camera_input("צלם את העלה לבדיקה", key="single_cam")
        else:
            img_file = st.file_uploader("בחר קובץ תמונה מהמחשב", type=['jpg', 'png', 'jpeg'], key="single_upload")
            
    with c2:
        if img_file:
            image = Image.open(img_file).convert('RGB')
            st.image(image, caption="התמונה שהוזנה לבדיקה", use_container_width=True)
            
            if model:
                with torch.no_grad():
                    output = model(transform(image).unsqueeze(0))
                    prob = torch.nn.functional.softmax(output[0], dim=0)
                    conf, pred = torch.max(prob, 0)
                
                if conf.item() < CONFIDENCE_THRESHOLD:
                    st.error("⚠️ לא זוהה עלה רלוונטי ברמת ודאות מספקת. אנא נסה לצלם שוב מקרוב ובתאורה טובה יותר.")
                else:
                    class_name = labels[pred.item()]
                    info = DISEASE_INFO.get(class_name, {"heb": class_name, "desc": "אין פירוט זמין", "tip": "אין המלצות זמינות"})
                    
                    st.markdown(f"### 🎯 אבחון מערכת: **{info['heb']}** (ביטחון: {conf.item()*100:.1f}%)")
                    
                    with st.container(border=True):
                        st.markdown(f"🔬 **פירוט המחלה והסימפטומים:**\n{info['desc']}")
                        st.divider()
                        st.markdown(f"💡 **המלצות לטיפול מעשי בשטח:**\n{info['tip']}")
            else:
                st.error("מודל הזיהוי אינו טעון בשרת.")

# ==========================================
# 📊 עמוד 2: מערכת הניסוי והטבלה המלאה
# ==========================================
elif st.session_state.page == "experiment_management":
    if st.button("🔙 חזרה לדף הבית", key="back_home_2"):
        st.session_state.page = "home"
        st.rerun()
        
    if plants_df is None:
        st.error("❌ קובץ הנתונים plants_experiment_73.csv חסר בשרת!")
        st.stop()
        
    st.header("📊 מערכת ניהול, ניטור וסנכרון ניסוי החיטה")
    st.divider()
    
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
        selected_label = st.selectbox("או בחר ישירות מהרשימה:", unique_plants, index=st.session_state.current_plant_idx, key="plant_selector")
        st.session_state.current_plant_idx = unique_plants.index(selected_label)
        
    plant_row = plants_df[plants_df['select_label'] == selected_label].iloc[0]
    plant_id = int(plant_row['id'])
    plant_name = str(plant_row['name'])
    
    plant_history = load_records_from_disk(plant_id)
    
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔢 מזהה ייחודי (ID)", str(plant_id))
        col2.metric("🌱 שם הצמח", plant_name)
        col3.metric("🧪 סוג טיפול", str(plant_row['#Treatment']))
        col4.metric("📉 מדד עקה (Stress)", f"{plant_row['stressDegree']:.3f}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 הערכה כללית ומצב צמח עדכני")
    treatment = plant_row['#Treatment']
    stress = plant_row['stressDegree']
    
    if treatment == 'Drought' and stress > 0.15:
        status_html = f"<div class='custom-card' style='border-right-color: #f44336;'>⚠️ <b>סטטוס פנוטיפי:</b> עקת יובש משמעותית (מדד: {stress:.3f}). הצמח מציג סימני מחסור חריפים במים.</div>"
    elif treatment == 'Drought':
        status_html = f"<div class='custom-card' style='border-right-color: #ff9800;'>🔸 <b>סטטוס פנוטיפי:</b> עקת יובש מתונה (מדד: {stress:.3f}). הצמח נמצא תחת מגבלת השקיה מבוקרת.</div>"
    else:
        status_html = f"<div class='custom-card'>✅ <b>סטטוס פנוטיפי:</b> תקין ויציב. קבוצת ביקורת (Control), משטר השקיה מלא.</div>"
    st.markdown(status_html, unsafe_allow_html=True)
    
    if len(plant_history) > 0:
        latest_record = plant_history[-1]
        c_name = latest_record.get("class_name", "")
        st.markdown(f"""
        <div class="custom-card" style="border-right-color: #2196f3; background-color: #e3f2fd;">
            🔍 <b>עדכון אבחון חזותי והמלצות טיפול ({latest_record['timestamp']}):</b><br><br>
            <b>🩺 מצב פתולוגי מאובחן:</b> {latest_record['diagnosis']}<br>
            <b>📝 תיאור מצב מהחממה:</b> {latest_record['notes']}
        </div>
        """, unsafe_allow_html=True)
        if c_name in DISEASE_INFO and c_name != "HealthyLeaf":
            with st.expander(f"🔬 לחץ לצפייה בפירוט מורחב והמלצות עבור {DISEASE_INFO[c_name]['heb']}", expanded=True):
                st.markdown(f"**תיאור המחלה:** {DISEASE_INFO[c_name]['desc']}")
                st.markdown(f"**🌱 המלצות לטיפול מעשי בשטח:** {DISEASE_INFO[c_name]['tip']}")
                
    st.divider()
    st.subheader("📊 נתוני הצמח המלאים מתוך הניסוי (תצורה אופקית)")
    all_cols = list(plants_df.columns)
    if 'id' in all_cols: all_cols.remove('id')
    if 'name' in all_cols: all_cols.remove('name')
    if 'select_label' in all_cols: all_cols.remove('select_label')
    ordered_cols = ['id', 'name'] + all_cols
    single_plant_df = plants_df[plants_df['id'] == plant_id][ordered_cols].copy()
    st.dataframe(single_plant_df, use_container_width=True, hide_index=True)
    
    st.divider()
    st.subheader("📸 בדיקה חזותית והוספה למאגר התיעודים")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            input_method = st.radio("בחר דרך להזנת תמונה:", ("צילום ישיר במצלמה 📸", "העלאת קובץ מהגלריה 📁"), key="exp_meth")
            if "מצלמה" in input_method:
                img_file = st.camera_input("צלם את העלה", key="exp_cam")
            else:
                img_file = st.file_uploader("בחר קובץ תמונה", type=['jpg', 'png', 'jpeg'], key="exp_upload")
        with c2:
            user_notes = st.text_area("✍️ תיאור ומצב הצמח בזמן הצילום:", placeholder="הקלד כאן תיאור מילולי של סימפטומים, גודל, צבע או הערות מיוחדות מהחממה...", height=150)
            
        if img_file:
            image = Image.open(img_file).convert('RGB')
            auto_diagnosis = "לא הופעל אבחון"
            class_name = "Unknown"
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
            
            if st.button(f"💾 שמור צילום ותיאור מצב למאגר המשותף", use_container_width=True):
                save_record_to_disk(plant_id, image, auto_diagnosis, user_notes if user_notes else "לא הוכנס פירוט חופשי", class_name)
                st.success("הנתונים נשמרו בהצלחה על השרת וזמינים כעת לכל המחשבים במערכת!")
                st.rerun()
                
    st.divider()
    st.subheader(f"🗄️ היסטוריית צילומים ותיעודים עבור צמח {plant_name} (מתוך מסד הנתונים הקבוע)")
    if len(plant_history) > 0:
        for record in reversed(plant_history):
            with st.container(border=True):
                hc1, hc2 = st.columns([1, 3])
                with hc1: st.image(record["image_path"], use_container_width=True)
                with hc2:
                    st.markdown(f"### 📅 תאריך ושעה: `{record['timestamp']}`")
                    st.markdown(f"**🔬 אבחון פתולוגי:** {record['diagnosis']}")
                    st.markdown(f"**📝 תיאור מצב שהוזן:** {record['notes']}")
                    c_ref = record.get("class_name", "")
                    if c_ref in DISEASE_INFO and c_ref != "HealthyLeaf":
                        st.info(f"💡 **הנחיית טיפול ארכיון:** {DISEASE_INFO[c_ref]['tip']}")
    else:
        st.info("אין עדיין צילומים מתועדים במאגר עבור צמח זה.")
