mod scanner;
mod model;
mod preprocessing;

use std::fs;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = &args[1];

    fs::create_dir_all("raw_jpgs").unwrap();
    fs::create_dir_all("fixed_jpgs").unwrap();

    let recovered = scanner::scan(path, 4096).unwrap();

    let mut model = model::Model::new("unet_model.onnx");

    for img_path in recovered{
        if !std::path::Path::new(&img_path).exists(){
            continue;
        }

        let tensor = preprocessing::jpeg_to_tensor(&img_path);

        let score = model.predict(tensor);

        if score > 0.5 {
            let file_name = std::path::Path::new(&img_path)
                .file_name()
                .unwrap()
                .to_str()
                .unwrap();

            let dest = format!("fixed_jpgs/{}", file_name);

            std::fs::copy(&img_path, &dest);
        }
    }
}
