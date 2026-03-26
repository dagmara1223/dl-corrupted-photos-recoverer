import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# disk paths
female_path = '/content/drive/MyDrive/dataset/ludzie/Female_Faces'
male_path = '/content/drive/MyDrive/dataset/ludzie/Male_Faces'

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

# mapping

NOISE_TYPES = {
    'gaussian':    lambda img, arr: add_gaussian_noise(arr, std=60),
    'salt_pepper': lambda img, arr: add_salt_and_pepper(arr, amount=0.15),
    'blur':        lambda img, arr: add_blur(img, radius=4),
    'bit_depth':   lambda img, arr: reduce_bit_depth(arr, bits=2),
    'bit_corrupt': lambda img, arr: add_random_bit_corruption(arr, corruption_rate=0.15),
    'patches':     lambda img, arr: add_random_patches(arr, n_patches=15),       
    'blobs':       lambda img, arr: add_blob_corruption(arr, n_blobs=8),         
    'combined':    lambda img, arr: add_combined_noise(img, arr),
}

def add_combined_noise(img, img_array):
    """blur -> gaussian -> salt&pepper -> bit depth -> bit corrupt -> patches -> blobs"""
    blurred = add_blur(img, radius=3)
    gaussed = add_gaussian_noise(blurred, std=45)
    salted = add_salt_and_pepper(gaussed, amount=0.10)
    quantized = reduce_bit_depth(salted, bits=2)
    corrupted = add_random_bit_corruption(quantized, corruption_rate=0.10)
    patched = add_random_patches(corrupted, n_patches=8)                       
    destroyed = add_blob_corruption(patched, n_blobs=5)                          
    return destroyed

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
        noisy_arr  = NOISE_TYPES[noise_type](img, arr)
        stem, ext  = os.path.splitext(fname)
        Image.fromarray(noisy_arr).save(os.path.join(output_path, f"{stem}__{noise_type}{ext}"))


# visualization

sample_path = os.path.join(female_path, os.listdir(female_path)[0])
sample_img  = Image.open(sample_path).convert('RGB')
sample_arr  = np.array(sample_img)

titles = ['Original', 'Gaussian', 'Salt & Pepper', 'Blur',
          'Bit Depth', 'Bit Corrupt', 'Patches', 'Blobs', 'Combined']
images = [
    sample_arr,
    add_gaussian_noise(sample_arr, 60),
    add_salt_and_pepper(sample_arr, 0.15),
    add_blur(sample_img, 4),
    reduce_bit_depth(sample_arr, bits=2),
    add_random_bit_corruption(sample_arr, 0.15),
    add_random_patches(sample_arr, n_patches=15),
    add_blob_corruption(sample_arr, n_blobs=8),
    add_combined_noise(sample_img, sample_arr),
]

# fig, axes = plt.subplots(1, 9, figsize=(36, 4))
# for ax, img_data, title in zip(axes, images, titles):
#     ax.imshow(img_data)
#     ax.set_title(title, fontsize=9)
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

def add_all_noise(img, img_array):
    """All types of noise on one picture"""
    arr = img_array.copy()
    arr = add_gaussian_noise(arr, std=60)
    arr = add_salt_and_pepper(arr, amount=0.15)
    # blur wymaga obiektu PIL Image, więc konwertujemy w locie
    arr = add_blur(Image.fromarray(arr), radius=4)
    arr = reduce_bit_depth(arr, bits=2)
    arr = add_random_bit_corruption(arr, corruption_rate=0.15)
    arr = add_random_patches(arr, n_patches=15)
    arr = add_blob_corruption(arr, n_blobs=8)
    return arr


full_destroyed = add_all_noise(sample_img, sample_arr)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(sample_arr); axes[0].set_title('Original', fontsize=13)
axes[1].imshow(full_destroyed); axes[1].set_title('Full noise', fontsize=13)
for ax in axes:
    ax.axis('off')

plt.suptitle('original vs full noise', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()
