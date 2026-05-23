use image::{imageops::FilterType, GenericImageView};

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