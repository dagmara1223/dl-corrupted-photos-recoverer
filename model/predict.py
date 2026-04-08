import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from architecture import JPGClassifier
import os

DEVICE = torch.device("cpu")
MODEL_PATH = "model/jpg_classifier.pth"

model = JPGClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def load_any_file_as_image(path):
    try:
        return Image.open(path).convert('RGB')
    except:
        with open(path, 'rb') as f:
            raw_data = f.read(256 * 256 * 3)
            raw_data += b'\x00' * (256 * 256 * 3 - len(raw_data))
            arr = np.frombuffer(raw_data[:256*256*3], dtype=np.uint8).reshape((256, 256, 3))
            return Image.fromarray(arr)

def predict_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    img = load_any_file_as_image(img_path)
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probability = output.item()
        
    result = "JPG" if probability > 0.5 else "SZUM/NON-JPG"
    print(f"Plik: {img_path}")
    print(f"Prawdopodobieństwo bycia JPG: {probability:.4f}")
    print(f"Werdykt: {result}\n")

if __name__ == "__main__":
    predict_image("model/train.py")
    predict_image("model/jpg_classifier.pth")
    predict_image(".gitignore")
    predict_image("README.md")
    # predict_image("moje zadania.txt")
    # predict_image("wykres_wiek.png")
    # predict_image("thesis.pdf")
    predict_image("dataset/ludzie/Noisy/test/non_jpg/noise_0.jpg")