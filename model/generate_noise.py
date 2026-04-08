import os
import numpy as np
from PIL import Image
import random

def file_to_image(file_path, output_path):
    try:
        # Czytamy plik jako surowe bajty
        with open(file_path, 'rb') as f:
            bytes_data = f.read(256 * 256 * 3)
            
        if len(bytes_data) < 256 * 256 * 3:
            # Jeśli plik za krótki, dopełniamy zerami
            bytes_data += b'\x00' * (256 * 256 * 3 - len(bytes_data))
        
        # Tworzymy obraz z bajtów
        arr = np.frombuffer(bytes_data[:256*256*3], dtype=np.uint8).reshape((256, 256, 3))
        img = Image.fromarray(arr)
        img.save(output_path)
        return True
    except:
        return False

def collect_garbage(base_dir, save_dir, count=200):
    os.makedirs(save_dir, exist_ok=True)
    found = 0
    # Przeszukujemy Twój projekt w poszukiwaniu plików .py, .venv, .git itp.
    for root, dirs, files in os.walk(base_dir):
        if "dataset" in root: continue # Nie chcemy brać zdjęć!
        for f in files:
            if found >= count: break
            full_path = os.path.join(root, f)
            if file_to_image(full_path, os.path.join(save_dir, f"sys_{found}.jpg")):
                found += 1
    print(f"Wygenerowano {found} próbek trudnego szumu systemowego.")

if __name__ == "__main__":
    collect_garbage(".", "dataset/ludzie/Noisy/train/non_jpg", count=300)