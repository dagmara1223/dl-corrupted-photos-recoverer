import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class JPGClassificationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        root_dir: ścieżka do folderu 'dataset/ludzie'
        split: 'train' lub 'test'
        """
        self.split = split
        self.transform = transform
        self.data = []

        noisy_root = os.path.join(root_dir, 'Noisy', split, 'noisy')
        for gender in ['Female_Faces', 'Male_Faces']:
            gender_path = os.path.join(noisy_root, gender)
            if os.path.exists(gender_path):
                for f in os.listdir(gender_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.data.append((os.path.join(gender_path, f), 1))

        non_jpg_path = os.path.join(root_dir, 'Noisy', split, 'non_jpg')
        if os.path.exists(non_jpg_path):
            for f in os.listdir(non_jpg_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.data.append((os.path.join(non_jpg_path, f), 0))

        if len(self.data) == 0:
            print(f"BŁĄD: Nie znaleziono żadnych danych w {root_dir}. Sprawdź ścieżki!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Problem z plikiem {img_path}: {e}")
            image = Image.new('RGB', (256, 256))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

if __name__ == "__main__":
    PATH = "dataset/ludzie" 
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    print("--- ROZPOCZYNAM TEST ŁADOWANIA DANYCH ---")
    
    try:
        dataset = JPGClassificationDataset(root_dir=PATH, split='train', transform=test_transform)
        
        print(f"Całkowita liczba znalezionych próbek: {len(dataset)}")
        
        if len(dataset) > 0:
            jpg_count = sum(1 for _, lbl in dataset.data if lbl == 1)
            noise_count = sum(1 for _, lbl in dataset.data if lbl == 0)
            
            print(f" - Klasa JPG: {jpg_count}")
            print(f" - Klasa Szum (non_jpg): {noise_count}")
            
            loader = DataLoader(dataset, batch_size=4, shuffle=True)
            images, labels = next(iter(loader))
            
            print("\nSukces! DataLoader działa.")
            print(f"Kształt paczki obrazów: {images.shape}") # Oczekiwane: [4, 3, 256, 256]
            print(f"Etykiety w paczce: {labels.tolist()}")
        else:
            print("\nUWAGA: Dataset jest pusty. Upewnij się, że foldery istnieją.")
            
    except Exception as e:
        print(f"\nWystąpił błąd podczas testu: {e}")
    
    print("--- KONIEC TESTU ---")