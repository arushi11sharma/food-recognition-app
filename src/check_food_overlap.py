import os
import pandas as pd

# -------------------------------
# Paths
# -------------------------------
selected_foods_path = "../data/selected_foods"
nutrition_csv_path = "/Users/arushi/food_recognition_app/Indian_Food_Nutrition_Processed.csv"

# -------------------------------
# 1️⃣ Get your selected food names
# -------------------------------
selected_foods = [d.lower().replace("_", " ") for d in os.listdir(selected_foods_path) if os.path.isdir(os.path.join(selected_foods_path, d))]

print("🍛 Foods in your selected_foods folder:")
print(selected_foods)
print("")

# -------------------------------
# 2️⃣ Load the nutrition dataset
# -------------------------------
df = pd.read_csv(nutrition_csv_path)
df.columns = df.columns.str.strip()  # remove spaces in column names
df.columns = df.columns.str.lower()  # lowercase all column names

# Confirm columns
print("🧾 Columns in nutrition dataset:")
print(df.columns.tolist())
print("")

# The main dish name column
dish_col = "dish name"

# Make a lowercase list of dishes in CSV
nutrition_foods = df[dish_col].str.lower().tolist()

print(f"🥗 Found {len(nutrition_foods)} dishes in the nutrition dataset.\n")

# -------------------------------
# 3️⃣ Check matches
# -------------------------------
matches = [food for food in selected_foods if any(food in n for n in nutrition_foods)]
missing = [food for food in selected_foods if food not in matches]

print("✅ Foods that match between your dataset and the Kaggle dataset:")
print(matches)
print("")
print("⚠️ Foods not found in Kaggle dataset:")
print(missing)

# -------------------------------
# 4️⃣ Save matched foods only
# -------------------------------
filtered_df = df[df[dish_col].str.lower().isin(matches)]
filtered_df.to_csv("../nutrition_lookup.csv", index=False)
print("\n💾 Saved matched foods to ../nutrition_lookup.csv")
