import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# disk paths
female_path = '/content/drive/MyDrive/dataset/ludzie/Female_Faces'
male_path   = '/content/drive/MyDrive/dataset/ludzie/Male_Faces'

output_base = '/content/drive/MyDrive/dataset/ludzie/Noisy'
os.makedirs(f'{output_base}/Female_Faces', exist_ok=True)
os.makedirs(f'{output_base}/Male_Faces',   exist_ok=True)

# noise generators

def add_gaussian_noise(img_array, std=60):
    noise = np.random.normal(0, std, img_array.shape)
    noisy = img_array.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_and_pepper(img_array, amount=0.15):
    noisy = img_array.copy()
    total = img_array.shape[0] * img_array.shape[1]
    n_salt = int(total * amount / 2)
    coords = [np.random.randint(0, i, n_salt) for i in img_array.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    n_pepper = int(total * amount / 2)
    coords = [np.random.randint(0, i, n_pepper) for i in img_array.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

def add_blur(img, radius=4):
    from PIL import ImageFilter
    return np.array(img.filter(ImageFilter.GaussianBlur(radius=radius)))

def reduce_bit_depth(img_array, bits=2):
    levels= 2 ** bits
    step = 256 // levels
    quantized = (np.floor(img_array.astype(np.float32) / step) * step).astype(np.uint8)
    return quantized

def add_random_bit_corruption(img_array, corruption_rate=0.15):
    flat = img_array.flatten().copy()
    n_corrupt = int(len(flat) * corruption_rate)
    corrupt_indices = np.random.choice(len(flat), n_corrupt, replace=False)
    bit_positions = np.random.randint(0, 8, n_corrupt)
    flat[corrupt_indices] ^= (1 << bit_positions).astype(np.uint8)
    return flat.reshape(img_array.shape)

def add_combined_noise(img, img_array):
    """blur → gaussian → salt&pepper → bit depth → bit corruption"""
    blurred = add_blur(img, radius=3)
    gaussed = add_gaussian_noise(blurred, std=45)
    salted  = add_salt_and_pepper(gaussed, amount=0.10)
    quantized = reduce_bit_depth(salted, bits=2)                         # ✅ fix 1
    destroyed = add_random_bit_corruption(quantized, corruption_rate=0.10)
    return destroyed

def add_random_patches(img_array, n_patches=15, patch_size_range=(20, 60)):
    noisy = img_array.copy()
    h, w  = img_array.shape[:2]

    for _ in range(n_patches):
        ph = random.randint(*patch_size_range)
        pw = random.randint(*patch_size_range)
        y  = random.randint(0, h - ph)
        x  = random.randint(0, w - pw)

        # random noise colors: white / black / gray / random color
        fill_type = random.choice(['white', 'black', 'gray', 'random'])
        if fill_type == 'white':
            fill = 255
        elif fill_type == 'black':
            fill = 0
        elif fill_type == 'gray':
            fill = random.randint(80, 180)
        else:
            fill = np.random.randint(0, 255, 3)

        noisy[y:y+ph, x:x+pw] = fill

    return noisy


def add_blob_corruption(img_array, n_blobs=8, max_radius=45):
    noisy  = img_array.copy().astype(np.float32)
    h, w = img_array.shape[:2]

    for _ in range(n_blobs):
        cy = random.randint(0, h)
        cx = random.randint(0, w)
        r  = random.randint(15, max_radius)

        Y, X   = np.ogrid[:h, :w]
        ry, rx = r * random.uniform(0.5, 1.5), r * random.uniform(0.5, 1.5)
        dist   = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2
        mask   = np.clip(1.0 - dist, 0, 1)

        fill = random.choice([0.0, 255.0, float(random.randint(60, 200))])

        for c in range(3):
            noisy[:, :, c] = noisy[:, :, c] * (1 - mask) + fill * mask

    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_digit_corruption(img_array, corruption_rate=0.15):
    """
    Randomly overwrites bytes with the number 0-9.
    Corruption_rate=0.15 ->15% of bytes are 0-9 instead of 0-255.

    Result: dark dots/areas because the values ​​0-9 are almost black.
    """
    flat = img_array.flatten().copy()
    n_corrupt = int(len(flat) * corruption_rate)

    corrupt_indices = np.random.choice(len(flat), n_corrupt, replace=False)
    flat[corrupt_indices] = np.random.randint(0, 10, n_corrupt).astype(np.uint8)

    return flat.reshape(img_array.shape)

# mapping

NOISE_TYPES = {
    'gaussian':    lambda img, arr: add_gaussian_noise(arr, std=60),
    'salt_pepper': lambda img, arr: add_salt_and_pepper(arr, amount=0.15),
    'blur':        lambda img, arr: add_blur(img, radius=4),
    'bit_depth':   lambda img, arr: reduce_bit_depth(arr, bits=2),
    'bit_corrupt': lambda img, arr: add_random_bit_corruption(arr, corruption_rate=0.15),
    'patches':     lambda img, arr: add_random_patches(arr, n_patches=15),
    'blobs':       lambda img, arr: add_blob_corruption(arr, n_blobs=8),
    'digit_corrupt':  lambda img, arr: add_digit_corruption(arr, corruption_rate=0.15),
    'combined':    lambda img, arr: add_combined_noise(img, arr),
}


#process

def process_dataset(input_path, output_path):
    files = [f for f in os.listdir(input_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    print(f"\nPrzetwarzam: {input_path}  ({len(files)} zdjęć)")
    for fname in tqdm(files):
        src  = os.path.join(input_path, fname)
        img  = Image.open(src).convert('RGB')
        arr  = np.array(img)
        noise_type = random.choice(list(NOISE_TYPES.keys()))
        noisy_arr = NOISE_TYPES[noise_type](img, arr)
        stem, ext = os.path.splitext(fname)
        Image.fromarray(noisy_arr).save(os.path.join(output_path, f"{stem}__{noise_type}{ext}"))


CORRUPTION_LEVELS = {
    'light': [                          #soft
        ('gaussian', 0.6),
        ('blur', 0.5),
        ('salt_pepper',0.3),
    ],
    'medium': [                         # medium
        ('gaussian', 0.7),
        ('blur', 0.5),
        ('salt_pepper', 0.5),
        ('bit_depth', 0.4),
        ('patches', 0.4),
    ],
    'heavy': [                          # rough
        ('gaussian', 0.9),
        ('salt_pepper', 0.8),
        ('blur', 0.7),
        ('bit_depth', 0.8),
        ('bit_corrupt', 0.7),
        ('patches', 0.7),
        ('blobs', 0.6),
        ('digit_corrupt', 0.5),
    ],
}

# levels
LEVEL_WEIGHTS = {'light': 0.35, 'medium': 0.40, 'heavy': 0.25}

# params per level
NOISE_PARAMS = {
    'light':  dict(gaussian_std=25,  blur_radius=2, sp_amount=0.05, bit_bits=4, patch_n=4,  blob_n=2,  corrupt_rate=0.05, digit_rate=0.05),
    'medium': dict(gaussian_std=45,  blur_radius=3, sp_amount=0.10, bit_bits=3, patch_n=8,  blob_n=5,  corrupt_rate=0.10, digit_rate=0.10),
    'heavy':  dict(gaussian_std=70,  blur_radius=5, sp_amount=0.18, bit_bits=2, patch_n=15, blob_n=10, corrupt_rate=0.20, digit_rate=0.20),
}

def apply_corruption_level(img, img_array, level='medium'):
    arr = img_array.copy()
    p = NOISE_PARAMS[level]
    funcs = CORRUPTION_LEVELS[level]

    for noise_type, prob in funcs:
        if random.random() > prob:
            continue

        if noise_type == 'gaussian':
            arr = add_gaussian_noise(arr, std=p['gaussian_std'])
        elif noise_type == 'blur':
            arr = add_blur(Image.fromarray(arr), radius=p['blur_radius'])
        elif noise_type == 'salt_pepper':
            arr = add_salt_and_pepper(arr, amount=p['sp_amount'])
        elif noise_type == 'bit_depth':
            arr = reduce_bit_depth(arr, bits=p['bit_bits'])
        elif noise_type == 'bit_corrupt':
            arr = add_random_bit_corruption(arr, corruption_rate=p['corrupt_rate'])
        elif noise_type == 'patches':
            arr = add_random_patches(arr, n_patches=p['patch_n'])
        elif noise_type == 'blobs':
            arr = add_blob_corruption(arr, n_blobs=p['blob_n'])
        elif noise_type == 'digit_corrupt':
            arr = add_digit_corruption(arr, corruption_rate=p['digit_rate'])

    return arr

# train test split save

from sklearn.model_selection import train_test_split

def process_dataset_split(input_path, gender, test_size=0.2):
    files = [f for f in os.listdir(input_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

    train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)

    for split, split_files in [('train', train_files), ('test', test_files)]:


        noisy_out = f'{output_base}/{split}/noisy/{gender}'
        clean_out = f'{output_base}/{split}/clean/{gender}'
        os.makedirs(noisy_out, exist_ok=True)
        os.makedirs(clean_out, exist_ok=True)

        print(f"\n[{split.upper()}] {gender}: {len(split_files)} zdjęć")

        for fname in tqdm(split_files):
            src = os.path.join(input_path, fname)
            img = Image.open(src).convert('RGB')
            arr = np.array(img)

            level = random.choices(
                list(LEVEL_WEIGHTS.keys()),
                weights=list(LEVEL_WEIGHTS.values())
            )[0]

            noisy_arr = apply_corruption_level(img, arr, level=level)

            stem, ext = os.path.splitext(fname)
            out_name  = f"{stem}__{level}{ext}"

            Image.fromarray(noisy_arr).save(os.path.join(noisy_out, out_name))
            Image.fromarray(arr).save(os.path.join(clean_out, out_name))   # ← ta sama nazwa!

process_dataset_split(female_path, 'Female_Faces')
process_dataset_split(male_path,   'Male_Faces')

print("Done.")

