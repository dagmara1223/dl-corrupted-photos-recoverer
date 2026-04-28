use std::fs::File;
use std::io::{BufReader, Read, Write};

const START_MARKER: [u8; 2] = [0xFF, 0xD8];
const END_MARKER: [u8; 2] = [0xFF, 0xD9];

const MAX_SIZE: usize = 20 * 1024 * 1024;

#[derive(PartialEq, Debug)]
enum State{
    SEARCHING,
    COLLECTING
}

pub fn scan(path: &str, chunk_size: usize) -> std::io::Result<()> {
    let image = File::open(path)?;
    let mut reader = BufReader::new(image);
    let mut buffer = vec![0u8; chunk_size];

    let mut last_byte = 0x00;
    let mut current_state: State = State::SEARCHING;

    let mut file_index = 0;
    let mut output: Option<File> = None;
    let mut current_size: usize = 0;

    let mut header_buffer: Vec<u8> = Vec::with_capacity(12);
    let mut saw_valid_segment = false;

    loop{
        let bytes_read = reader.read(&mut buffer)?;

        if bytes_read == 0{
            break;
        }

        process_chunk(&buffer[..bytes_read], last_byte, &mut current_state, &mut output, &mut file_index, &mut current_size, &mut header_buffer, &mut saw_valid_segment);
        last_byte = buffer[bytes_read - 1];
    }
    Ok(())
}

fn process_chunk(chunk: &[u8], last_byte: u8, current_state: &mut State, output: &mut Option<File>, file_index: &mut usize, current_size: &mut usize,
header_buffer: &mut Vec<u8>, saw_valid_segment: &mut bool) {
    let mut skip_next = false;

    if compare_to_marker(last_byte, chunk[0], current_state, output, file_index, current_size, header_buffer, saw_valid_segment) {
        println!("{:?}, Chunk at boundary", current_state);
    }

    for i in 0..chunk.len().saturating_sub(1) {
        if skip_next {
            skip_next = false;
            continue;
        }

        let x = chunk[i];
        let y = chunk[i + 1];

        let is_marker = compare_to_marker(x, y, current_state, output, file_index, current_size, header_buffer, saw_valid_segment);

        if is_marker{
            skip_next = true;
        }

        if *current_state == State::COLLECTING && !is_marker && !skip_next{
            if header_buffer.len() < 12 {
                header_buffer.push(x);

                if header_buffer.len() == 12 && !validate_header(header_buffer) {
                    let _ = std::fs::remove_file(format!("image_{:04}.jpg", file_index));
                    *output = None;
                    *current_state = State::SEARCHING;
                    *current_size = 0;
                    header_buffer.clear();
                    return;
                }
            }

            if x == 0xFF && (y == 0xE0 || y == 0xE1) {
                *saw_valid_segment = true;
            }

            if let Some(file) = output.as_mut() {

                if check_max_size(*current_size) {
                    let _ = std::fs::remove_file(format!("image_{:04}.jpg", file_index));
                    *output = None;
                    *current_state = State::SEARCHING;
                    *current_size = 0;
                    return;
                }
                let _ = file.write_all(&[x]);
                *current_size += 1;
            }
        }
    }

    let last = chunk[chunk.len() - 1];

    if *current_state == State::COLLECTING {
        if x == 0xFF && (y == 0xE0 || y == 0xE1) {
                *saw_valid_segment = true;
        }

        if header_buffer.len() < 12 {
            header_buffer.push(last);

            if header_buffer.len() == 12 && !validate_header(header_buffer) {
                let _ = std::fs::remove_file(format!("image_{:04}.jpg", file_index));
                *output = None;
                *current_state = State::SEARCHING;
                *current_size = 0;
                header_buffer.clear();
                return;
            }
        }

        if let Some(file) = output.as_mut() {
            if check_max_size(*current_size) {
                let _ = std::fs::remove_file(format!("image_{:04}.jpg", file_index));
                *output = None;
                *current_state = State::SEARCHING;
                *current_size = 0;
                return;
            }
            let _ = file.write_all(&[last]);
            *current_size += 1;
        }
    }
}

fn compare_to_marker(x: u8, y: u8, current_state: &mut State, output: &mut Option<File>, file_index: &mut usize, 
    current_size: &mut usize, header_buffer: &mut Vec<u8>, saw_valid_segment: &mut bool) -> bool {
    if *current_state == State::SEARCHING {
        if x == START_MARKER[0] && y == START_MARKER[1] {
            *current_state = State::COLLECTING;
            header_buffer.clear();
            saw_valid_segment = false;

            *file_index += 1;

            let filename = format!("image_{:04}.jpg", *file_index);
            *output = Some(File::create(filename).unwrap());

            if let Some(file) = output.as_mut() {
                let _ = file.write_all(&START_MARKER);
                *current_size += 1;
            }

            return true;
        }
    } else {
        if x == END_MARKER[0] && y == END_MARKER[1] {
            *current_state = State::SEARCHING;

            if !*saw_valid_segment {
                let _ = std::fs::remove_file(format!("image_{:04}.jpg", file_index));
                *output = None;
                *current_state = State::SEARCHING;
                *current_size = 0;
                return true;
            }

            if let Some(file) = output.as_mut() {
                let _ = file.write_all(&END_MARKER);
                *current_size += 1;
            }

            *output = None;
            return true;
        }
    }

    false
}

fn check_max_size(current_size: usize) -> bool {
    current_size >= MAX_SIZE
}

fn validate_header(header: &[u8]) -> bool {
    if header.len() < 12 {
        return false;
    }

    let is_jfif = &header[6..10] == b"JFIF";
    let is_exif = header[6..12].starts_with(b"Exif");

    is_jfif || is_exif
}