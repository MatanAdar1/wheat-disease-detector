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

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        direction: rtl !important;
        text-align: right !important;
    }
    h1, h2, h3, h4, p, label, .stMarkdown, [data-testid="stWidgetLabel"] {
        direction: rtl !important;
        text-align: right !important;
    }
    div[role="radiogroup"] {
        direction: rtl !important;
        text-align: right !important;
    }
    [data-testid="stFileUploader"] {
        direction: rtl !important;
        text-align: right !important;
    }
    .stMetric {
        text-align: right !important;
    }
    </style>
    """, unsafe_allow_html=True)

FILE_ID = '161ysydHCyvLOoVWkwWqJT5RpcMn_0rVu'
MODEL_PATH = 'best_resnet18_wheat.pt'
CONFIDENCE_THRESHOLD = 0.25

if "plant_history" not in st.session_state:
    st.session_state.plant_history = {}

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
        st.error(f"שגיאה בטעינת המודל: {e}")
        return None, None

plants_df = load_experiment_data()

if plants_df is None:
    st.title("מערכת חכמה לניהול ומאגר ניסוי חיטה 🌾")
    st.error("❌ קובץ הנתונים plants_experiment_73.csv חסר בשרת!")
    st.info("אנא ודאו שהעלתם את הקובץ לתיקייה הראשית ב-GitHub לצד קובץ ה-app.py.")
    st.stop()

model, labels = load_wheat_model()

st.title("מערכת חכמה לניהול, ניטור וזיהוי מחלות חיטה 🌾")
st.write("מבצעים: נבו הלר ומתן אדר | מנחה: אסי ברק")
st.divider()

st.sidebar.header("🕹️ תפריט ניווט ובחירה")
plants_df['select_label'] = plants_df.apply(lambda r: f"{r['name']} (ID: {r['id']})", axis=1)
selected_label = st.sidebar.selectbox("בחר צמח לפי שם ומזהה:", plants_df['select_label'].unique())

plant_row = plants_df[plants_df['select_label'] == selected_label].iloc[0]
plant_id = int(plant_row['id'])
plant_name = str(plant_row['name'])

col1, col2, col3, col4 = st.columns(4)
col1.metric("מזהה ייחודי (ID)", str(plant_id))
col2.metric("שם הצמח", plant_name))
col3.metric("סוג טיפול", str(plant_row['#Treatment']))
col4.metric("מדד עקה", f"{plant_row['stressDegree']:.3f}")

st.subheader("📊 נתוני הצמח המלאים מתוך הניסוי")
clean_row = plant_row.drop('select_label')
display_df = pd.DataFrame({
    "פרמטר / מדד": clean_row.index,
    "ערך מוקלט": clean_row.values
}).astype(str)

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("📸 תיעוד חזותי והוספה למאגר הצמח")

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

c1, c2 = st.columns(2)

with c1:
    input_method = st.radio("בחר דרך להזנת תמונה למאגר צמח זה:", 
                            ("צילום ישיר במצלמה 📸", "העלאת קובץ מהגלריה 📁"), key="input_meth")
    if "מצלמה" in input_method:
        img_file = st.camera_input("צלם את העלה", key="capture_photo")
    else:
        img_file = st.file_uploader("בחר קובץ תמונה", type=['jpg', 'png', 'jpeg'], key="upload_photo")

with c2:
    user_notes = st.text_area("✍️ פירוט על מצב הצמח באותו זמן:", 
                              placeholder="הקלד כאן תיאור מילולי, תצפיות מיוחדות או הערות מהשטח...", height=150)

if img_file:
    image = Image.open(img_file).convert('RGB')
    
    auto_diagnosis = "לא הופעל אבחון"
    if model:
        with torch.no_grad():
            output = model(transform(image).unsqueeze(0))
            prob = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(prob, 0)
        
        if conf.item() < CONFIDENCE_THRESHOLD:
            auto_diagnosis = "לא זוהה עלה רלוונטי בתמונה"
        else:
            class_name = labels[pred.item()]
            auto_diagnosis = DISEASE_INFO.get(class_name, {"heb": class_name})["heb"]
    
    st.write("---")
    st.markdown(f"**אבחון אוטומטי זמני:** {auto_diagnosis}")
    
    if st.button(f"💾 שמור תיעוד זה למאגר של צמח {plant_name}"):
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        if plant_id not in st.session_state.plant_history:
            st.session_state.plant_history[plant_id] = []
            
        st.session_state.plant_history[plant_id].append({
            "timestamp": current_time,
            "image": image,
            "notes": user_notes if user_notes else "לא הוכנס פירוט חופשי",
            "diagnosis": auto_diagnosis
        })
        st.success(f"התיעוד נשמר בהצלחה במאגר של צמח {plant_name}!")
        st.rerun()

st.divider()
st.subheader(f"🗄️ מאגר תמונות והיסטוריית תיעודים - צמח {plant_name}")

if plant_id in st.session_state.plant_history and len(st.session_state.plant_history[plant_id]) > 0:
    history_list = st.session_state.plant_history[plant_id]
    
    for idx, record in enumerate(reversed(history_list)):
        with st.container():
            hc1, hc2 = st.columns([1, 3])
            with hc1:
                st.image(record["image"], use_container_width=True)
            with hc2:
                st.markdown(f"### 📅 תאריך ושעה: `{record['timestamp']}`")
                st.markdown(f"**🔬 אבחון מערכת:** {record['diagnosis']}")
                st.markdown(f"**📝 פירוט מצב הצמח:** {record['notes']}")
            st.write("---")
else:
    st.info("לא קיימים תיעודים או צילומים במאגר עבור צמח זה עדיין. השתמשו בממשק מעל כדי להוסיף את הצילום הראשון.")
