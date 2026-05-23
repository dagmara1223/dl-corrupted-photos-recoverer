use ort::session::Session;
use ort::value::Value;
use ndarray::Array;

pub struct Model {
    session: Session,
}

impl Model {
    pub fn new(model_path: &str) -> Self {
        let session = Session::builder()
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();

        Self { session }
    }

    pub fn restore(&mut self, input: Vec<f32>) -> Vec<f32> {
        let input = Array::from_shape_vec((1, 3, 256, 256), input).unwrap();

        let input = Value::from_array(input).unwrap();

        let outputs = self.session.run(vec![("input_image", input),]).unwrap();

        let out = outputs[0].try_extract_tensor::<f32>().unwrap();

        out.1.to_vec()
    }
}