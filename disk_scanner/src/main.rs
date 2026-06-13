mod scanner;
mod model;
mod processing;

use std::fs;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = &args[1];

    fs::create_dir_all("raw_jpgs").unwrap();
    fs::create_dir_all("fixed_jpgs").unwrap();

    let recovered = scanner::scan(path, 4096).unwrap();

    let mut model = model::Model::new("unet_final.onnx");

    for img_path in recovered{
        if !std::path::Path::new(&img_path).exists(){
            continue;
        }

        let input = match std::panic::catch_unwind(|| {
            processing::jpeg_to_tensor(&img_path)
        }) {
            Ok(v) => v,
            Err(_) => {
                println!("Panic while processing image: {}", img_path);
                continue;
            }
        };

        let restored = model.restore(input.data);

        let file_name = std::path::Path::new(&img_path).file_name().unwrap().to_str().unwrap();

        let dest = format!("fixed_jpgs/{}", file_name);

        processing::tensor_to_image(restored, &dest, input.width, input.height);
    }
    println!("Done")
}
