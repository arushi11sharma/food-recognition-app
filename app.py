import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from difflib import get_close_matches
import streamlit.components.v1 as components

# -----------------------------
# 🧠 Helper Functions
# -----------------------------
def normalize_name(name):
    return name.lower().replace("_", " ").replace("-", " ").strip()

def find_best_match(food_name, df):
    clean_name = normalize_name(food_name)
    dish_names = df["Dish name"].apply(normalize_name).tolist()
    matches = get_close_matches(clean_name, dish_names, n=1, cutoff=0.6)
    if matches:
        matched_name = matches[0]
        return df[df["Dish name"].apply(normalize_name) == matched_name]
    else:
        return pd.DataFrame()

# -----------------------------
# 🔧 Load Model
# -----------------------------
checkpoint = torch.load("models/food_classifier.pt", map_location=torch.device('cpu'))
class_to_idx = checkpoint['class_to_idx']
num_classes = len(class_to_idx)

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
idx_to_class = {v: k for k, v in class_to_idx.items()}

# -----------------------------
# Load Nutrition Data
# -----------------------------
nutrition_df = pd.read_csv("nutrition_lookup.csv")

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="FoodVision", layout="centered")

# -----------------------------
# 🌙 Dark Theme Styling
# -----------------------------
st.markdown("""
<style>
/* ----------------- Global Page Styles ----------------- */
.stApp {
    background-color: #0f1116 !important;  /* Deep dark background */
    color: #ffffff !important;             /* White text */
    font-family: 'Poppins', sans-serif;
}

/* ----------------- Headings ----------------- */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* ----------------- Text Elements ----------------- */
p, span, div, label {
    color: #eaeaea !important;
}

/* ----------------- File Uploader ----------------- */
div[data-testid="stFileUploader"] > section {
    background-color: #1c1e24 !important;
    border: 1.5px solid #B6D6F2 !important;
    border-radius: 10px !important;
    padding: 1rem;
    color: #ffffff !important;
}

/* ----------------- Dropdown (Closed State) ----------------- */
div[data-baseweb="select"] > div {
    background-color: #1c1e24 !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #B6D6F2 !important;
}

/* ----------------- Buttons ----------------- */
div.stButton > button {
    background-color: #B6D6F2 !important;
    color: #0f1116 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s ease-in-out;
}
div.stButton > button:hover {
    background-color: #A3C7E4 !important;
    transform: scale(1.03);
}

/* ----------------- Sliders ----------------- */
[data-testid="stSlider"] > div > div > div > div {
    background-color: #B6D6F2 !important;  /* Track color */
}
[data-testid="stSliderThumb"] {
    background-color: #B6D6F2 !important;  /* Handle color */
    border: 2px solid #A3C7E4 !important;
}

/* ----------------- Data Table ----------------- */
thead tr th {
    background-color: #1c1e24 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-bottom: 1px solid #B6D6F2 !important;
}
tbody tr {
    background-color: #14161b !important;
    color: #ffffff !important;
}
tbody tr:hover {
    background-color: #1f2229 !important;
}
table {
    border-radius: 10px !important;
    border: 1px solid #B6D6F2 !important;
}

/* ----------------- Progress Bar ----------------- */
[data-testid="stProgress"] > div > div {
    background-color: #B6D6F2 !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1 style='text-align:center;'>FoodVision</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#B6D6F2;'>A food recognition and nutrition tracker</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("📸 Upload your food image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if "meal_log" not in st.session_state:
    st.session_state.meal_log = []

# -----------------------------
# 🔍 Prediction Function
# -----------------------------
def predict_food(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    predicted_class = idx_to_class[predicted.item()]
    return predicted_class, confidence.item() * 100

# -----------------------------
# 📸 Process Uploaded Image
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    predicted_class, confidence = predict_food(image)
    display_name = normalize_name(predicted_class).title()
    st.subheader(f"🍽️ {display_name} ({confidence:.2f}% confidence)")

    row = find_best_match(predicted_class, nutrition_df)
    if not row.empty:
        r = row.iloc[0]

        if normalize_name(predicted_class) == "biryani":
            st.write("### 🍛 Select Biryani Type")
            biryani_type = st.selectbox("Choose the type of Biryani:",
                                        ["Veg Biryani", "Chicken Biryani", "Mutton Biryani"])

            base_cal = r["Calories (kcal)"]
            base_protein = r["Protein (g)"]
            base_fats = r["Fats (g)"]
            base_carbs = r["Carbohydrates (g)"]

            if biryani_type == "Veg Biryani":
                cal_adj, prot_adj, fat_adj, carb_adj = 0, 0, 0, 0
            elif biryani_type == "Chicken Biryani":
                cal_adj, prot_adj, fat_adj, carb_adj = 80, 6, 4, -2
            else:
                cal_adj, prot_adj, fat_adj, carb_adj = 150, 9, 8, -3

            r["Calories (kcal)"] = base_cal + cal_adj
            r["Protein (g)"] = base_protein + prot_adj
            r["Fats (g)"] = base_fats + fat_adj
            r["Carbohydrates (g)"] = base_carbs + carb_adj

            st.success(f"You selected: **{biryani_type}**")
            st.dataframe(r.to_frame().T, use_container_width=True)
        else:
            st.write("### Nutritional Info (per serving)")
            st.dataframe(r.to_frame().T, use_container_width=True)

        st.write("### Customize Ingredients")
        oil_type = st.selectbox("Type of Oil", ["None", "Olive Oil", "Sunflower Oil", "Ghee", "Butter"])
        extra_oil = st.slider("Extra Oil (tsp)", 0, 5, 0)
        sugar_type = st.selectbox("Type of Sweetener", ["None", "White Sugar", "Brown Sugar", "Jaggery", "Honey"])
        extra_sugar = st.slider("Extra Sweetener (tsp)", 0, 5, 0)

        oil_calories = {"None": 0, "Olive Oil": 40, "Sunflower Oil": 45, "Ghee": 50, "Butter": 45}
        sugar_calories = {"None": 0, "White Sugar": 16, "Brown Sugar": 15, "Jaggery": 13, "Honey": 21}

        calorie_adjustment = (extra_oil * oil_calories[oil_type]) + (extra_sugar * sugar_calories[sugar_type])
        adjusted_calories = r["Calories (kcal)"] + calorie_adjustment

        st.success(f"Adjusted Calories: {adjusted_calories:.0f} kcal")

        if oil_type in ["Ghee", "Butter"]:
            st.info("💡 Try **Olive Oil** for a lighter, heart-friendly meal.")
        if sugar_type in ["White Sugar", "Jaggery"]:
            st.info("💡 Switch to **Honey** for a natural alternative.")

        if st.button("➕ Add to Meal Log"):
            log_entry = {
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Dish": display_name,
                "Calories": adjusted_calories,
                "Oil": oil_type,
                "Sugar": sugar_type
            }
            st.session_state.meal_log.append(log_entry)
            st.success(f"✅ {display_name} added to your log!")
    else:
        st.warning("⚠️ Nutrition info not found for this dish.")

# -----------------------------
# 📅 Meal Log + Donut Chart
# -----------------------------
if st.session_state.meal_log:
    st.markdown("---")
    st.subheader("Today's Meal Log")
    df_log = pd.DataFrame(st.session_state.meal_log)
    st.dataframe(df_log, use_container_width=True)

    total_calories = sum(item["Calories"] for item in st.session_state.meal_log)
    calorie_goal = st.slider("🎯 Daily Calorie Goal", 1200, 3000, 2000)
    progress = min(total_calories / calorie_goal, 1.0)

    st.markdown(f"### Total: **{total_calories:.0f} / {calorie_goal} kcal**")
    st.progress(progress)

    remaining = max(calorie_goal - total_calories, 0)
    fig, ax = plt.subplots(figsize=(2.8, 2.8), facecolor="#0f1116")
    ax.pie(
        [total_calories, remaining],
        labels=['Consumed', 'Remaining'],
        startangle=90,
        counterclock=False,
        autopct='%1.0f%%',
        colors=['#B6D6F2', '#1c1e24'],  # pastel blue + dark background
        textprops={'color': 'white', 'fontsize': 10},
        wedgeprops=dict(width=0.4)
    )
    ax.set_aspect('equal')
    plt.setp(ax.texts, color="white")
    st.pyplot(fig)

    if total_calories >= calorie_goal:
        st.success("🎉 You’ve hit your daily calorie goal!")
    else:
        st.info(f"You can still enjoy about **{remaining:.0f} kcal** today.")
