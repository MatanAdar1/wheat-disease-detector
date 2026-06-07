import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown
import pandas as pd

st.set_page_config(page_title="ניהול וניטור ניסוי חיטה 🌾", page_icon="🌾", layout="wide")

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
    plants_df = pd.read_csv("plants_experiment_73.csv")
    temp_df = pd.read_csv("Temperature_0_5TM__GraphViewer.csv")
    return plants_df, temp_df

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
        st.error(f"שגיאה בטעינת המודל: {e}")
        return None, None

plants_df, temp_df = load_experiment_data()
model, labels = load_wheat_model()

st.title("מערכת חכמה לניהול, ניטור וזיהוי מחלות חיטה 🌾")
st.write("מבצעים: נבו הלר ומתן אדר | מנחה: אסי ברק")
st.divider()

st.sidebar.header("סינון ובחירת צמח")
selected_plant = st.sidebar.selectbox("בחר מזהה צמח מתוך הניסוי:", plants_df['name'].unique())

plant_row = plants_df[plants_df['name'] == selected_plant].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("סוג טיפול", str(plant_row['#Treatment']))
col2.metric("קו גנטי (#line)", str(plant_row['#line']))
col3.metric("מדד עקה (Stress Degree)", f"{plant_row['stressDegree']:.3f}")
col4.metric("קצב התאוששות", f"{plant_row['resilienceRate']:.3f}")

st.subheader("📋 הערכה כללית של מצב הצמח")
treatment = plant_row['#Treatment']
stress = plant_row['stressDegree']

if treatment == 'Drought' and stress > 0.15:
    st.error(f"**סטטוס: עקת יובש משמעותית.** הצמח משויך לקבוצת הטיפול ביתר יובש ומציג מדד לחץ גבוה ({stress:.3f}). מומלץ לבצע בדיקה חזותית קרובה כדי לשלול התפתחות מחלות משניות המנצלות את חולשת הצמח.")
elif treatment == 'Drought':
    st.warning(f"**סטטוס: עקת יובש מתונה.** הצמח נמצא תחת מגבלת מים אך מראה יציבות ומדדי הסתגלות תקינים בשלב זה.")
else:
    st.success("**סטטוס: תקין ויציב.** הצמח משויך לקבוצת הביקורת (Control), תנאי הלחות אופטימליים והתפתחות המסה שלו מתנהלת כמצופה.")

st.divider()
st.subheader("📈 נתוני טמפרטורה לאורך זמן (°C)")

temp_col_name = f"{selected_plant} - Temperature/0/5TM (°c)"
if temp_col_name in temp_df.columns:
    plot_data = temp_df[['Timestamp', temp_col_name]].dropna()
    if not plot_data.empty:
        plot_data['Timestamp'] = pd.to_datetime(plot_data['Timestamp'])
        plot_data = plot_data.set_index('Timestamp')
        plot_data = plot_data.rename(columns={temp_col_name: "טמפרטורה"})
        st.line_chart(plot_data)
    else:
        st.info("אין נקודות נתונים זמינות להצגה בגרף עבור צמח זה.")
else:
    st.info("לא נמצאה עמודת טמפרטורה תואמת עבור הצמח הנבחר בקובץ הרצף העתידי.")

st.divider()
st.subheader("📸 בדיקה חזותית וזיהוי מחלות בזמן אמת")

with st.expander("לחץ כאן כדי לפתוח את ממשק הצילום והאבחון של העלה"):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_method = st.radio("בחר כיצד להזין תמונה לבדיקה:", 
                            ("צילום במצלמה 📸", "העלאת תמונה מהגלריה 📁"), key="visual_inspect")

    if "מצלמה" in input_method:
        img_file = st.camera_input("צלם את העלה", key="cam_key")
    else:
        img_file = st.file_uploader("בחר קובץ תמונה (JPG, PNG, JPEG)", type=['jpg', 'png', 'jpeg'], key="file_key")

    if img_file and model:
        image = Image.open(img_file).convert('RGB')
        st.image(image, caption=f"תמונת עלה שנבחנה עבור צמח {selected_plant}", use_container_width=True)
        
        with torch.no_grad():
            output = model(transform(image).unsqueeze(0))
            prob = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(prob, 0)

        if conf.item() < CONFIDENCE_THRESHOLD:
            st.markdown("<br>", unsafe_allow_html=True)
            st.warning("לא זוהה עלה בתמונה, נסה לצלם שוב")
        else:
            class_name = labels[pred.item()]
            info = DISEASE_INFO.get(class_name, {"heb": class_name, "desc": "", "tip": ""})

            st.divider()
            color = "green" if "Healthy" in class_name else "red"
            st.markdown(f"### אבחנה עבור {selected_plant}: :{color}[{info['heb']}]")
            st.write(f"**תיאור המחלה:** {info['desc']}")
            st.info(f"**המלצה לטיפול:** {info['tip']}")
