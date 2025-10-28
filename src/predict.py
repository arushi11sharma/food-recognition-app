import torch
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd

# -------------------------------
# 1. Load the trained model
# -------------------------------

checkpoint_path = "../models/food_classifier.pt"

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
class_to_idx = checkpoint['class_to_idx']
num_classes = len(class_to_idx)

# Initialize the model
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Replace the classifier to match trained classes
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, num_classes)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Reverse class index mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}

print("✅ Model loaded successfully with classes:")
print(list(idx_to_class.values()))

# -----------------------------
# 2. Load Nutrition Data
# -----------------------------

nutrition_df = pd.read_csv("../nutrition_lookup.csv")

def get_nutrition(food_name):
    """Retrieve and display nutrition info for a given food."""
    row = nutrition_df[nutrition_df["Dish name"].str.lower() == food_name.lower()]
    if not row.empty:
        r = row.iloc[0]
        print(f"\n🥗 Nutritional Info for {food_name.title()} (per {r['Serving Size (g)']}g):")
        print(f"Calories: {r['Calories (kcal)']} kcal")
        print(f"Protein: {r['Protein (g)']} g")
        print(f"Carbohydrates: {r['Carbohydrates (g)']} g")
        print(f"Fats: {r['Fats (g)']} g")
        print(f"Fibre: {r['Fibre (g)']} g")
        print(f"Vitamin C: {r['Vitamin C (mg)']} mg")
    else:
        print(f"⚠️ No nutrition info found for '{food_name}'")

# -------------------------------
# 3. Define image transform
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# 4. Prediction function
# -------------------------------
def predict_food(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = idx_to_class[predicted.item()]
    print(f"\n🍽️ Predicted Food: {predicted_class} ({confidence.item() * 100:.2f}% confidence)")

    # ✅ Show nutrition info for the predicted food
    get_nutrition(predicted_class)

# -------------------------------
# 5. Test the prediction
# -------------------------------
test_image = "../data/selected_foods/bhatura/0b3ed8d096.jpg"
predict_food(test_image)
