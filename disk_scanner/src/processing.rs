use image::{imageops::FilterType, GenericImageView};
use image::{ImageBuffer, RgbImage};

pub fn jpeg_to_tensor(path: &str) -> Vec<f32> {
    let img = image::open(path)
        .expect("Invalid image")
        .to_rgb8();

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

    tensor
}

pub fn tensor_to_image(tensor: Vec<f32>, path: &str){
    let width = 256;
    let height = 256;
    let channels = 3;

    assert_eq!(tensor.len(), width * height * channels);

    let mut img: Vec<u8> = vec![0; width * height * channels];

    let hw = width * height;

    for c in 0..channels {
        for i in 0..hw {
            let v = tensor[c * hw + i];

            let v = (v * 255.0).clamp(0.0, 255.0) as u8;

            img[i * 3 + c] = v;
        }
    }

    let img = RgbImage::from_raw(width as u32, height as u32, img)
        .expect("invalid image buffer");

    img.save(path).unwrap();
}