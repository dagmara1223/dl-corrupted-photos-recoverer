use image::{imageops::FilterType, GenericImageView};
use image::{ImageBuffer, RgbImage};

pub struct ImageTensor {
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

pub fn jpeg_to_tensor(path: &str) -> ImageTensor {
    let img = image::open(path).expect("Invalid image").to_rgb8();

    let (w, h) = img.dimensions();

    let resized = image::imageops::resize(&img, 256, 256, FilterType::Triangle);

    let mut tensor = Vec::with_capacity(3 * 256 * 256);

    for c in 0..3 {
        for y in 0..256 {
            for x in 0..256 {
                let pixel = resized.get_pixel(x, y);
                tensor.push(pixel[c] as f32 / 255.0);
            }
        }
    }

    ImageTensor {
        data: tensor,
        width: w,
        height: h,
    }
}

pub fn tensor_to_image(tensor: Vec<f32>, path: &str, width: u32, height: u32){
    let hw = 256 * 256;
    
    let mut img: Vec<u8> = vec![0; hw * 3];
    assert_eq!(tensor.len(), hw * 3, "unexpected tensor size");

    for c in 0..3 {
        for i in 0..hw {
            let v = tensor[c * hw + i];

            let v = (v * 255.0).clamp(0.0, 255.0) as u8;

            img[i * 3 + c] = v;
        }
    }

    let img = RgbImage::from_raw(256 as u32, 256 as u32, img)
        .expect("invalid image buffer");

    let resized = image::imageops::resize(
        &img,
        width,
        height,
        FilterType::Triangle,
    );

    resized.save(path).unwrap();
}