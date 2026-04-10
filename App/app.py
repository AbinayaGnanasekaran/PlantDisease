import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd

# -----------------------------
# Load CSV Files
# -----------------------------
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# -----------------------------
# Temporary Model Definition
# -----------------------------
class TempModel(nn.Module):
    def __init__(self, num_classes=39):
        super(TempModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, (3, 3))
        self.fc = nn.Linear(5 * 222 * 222, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -----------------------------
# Load the Model
# -----------------------------
model_path = "ResNet50.pt"

if not os.path.exists(model_path):
    print("ERROR: Model file missing:", model_path)

model = TempModel(num_classes=len(disease_info))

try:
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    print("Model loaded successfully.")
except Exception as e:
    print("Model load error:", e)

model.eval()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    return pred

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':

        upload_path = "static/uploads"
        os.makedirs(upload_path, exist_ok=True)

        image = request.files['image']
        file_path = os.path.join(upload_path, image.filename)
        image.save(file_path)

        pred = predict(file_path)

        title = disease_info.loc[pred, 'disease_name']
        description = disease_info.loc[pred, 'description']
        prevent = disease_info.loc[pred, 'Possible Steps']
        image_url = disease_info.loc[pred, 'image_url']

        supplement_name = supplement_info.loc[pred, 'supplement name']
        supplement_image = supplement_info.loc[pred, 'supplement image']
        supplement_buy_link = supplement_info.loc[pred, 'buy link']

        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image,
            buy_link=supplement_buy_link
        )

    return redirect('/')

@app.route('/market')
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

if __name__ == '__main__':
    app.run(debug=True)
