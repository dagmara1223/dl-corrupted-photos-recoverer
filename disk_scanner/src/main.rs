mod scanner;
mod model;

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = &args[1];
    let scan_result = scanner::scan(path, 4096);
}
