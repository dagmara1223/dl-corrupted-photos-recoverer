use std::fs::File;
use std::io::{BufReader, Read};

pub fn scan(path: &str, chunk_size: usize) -> std::io::Result<()> {
    let image = File::open(path)?;
    let mut reader = BufReader::new(image);
    let mut buffer = vec![0u8; chunk_size];

    loop{
        let bytes_read = reader.read(&mut buffer)?;

        if bytes_read == 0{
            break;
        }

        println!("{:02x?}", &buffer[..16]);
    }
    Ok(())
}
