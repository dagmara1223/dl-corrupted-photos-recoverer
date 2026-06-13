use std::fs::File;
use std::io::{BufReader, Read, Write};

const START_MARKER: [u8; 2] = [0xFF, 0xD8];
const END_MARKER: [u8; 2] = [0xFF, 0xD9];

const MAX_SIZE: usize = 20 * 1024 * 1024;

#[derive(PartialEq, Debug)]
enum State {
    SEARCHING,
    COLLECTING,
}

pub fn scan(path: &str, chunk_size: usize) -> std::io::Result<Vec<String>> {
    let mut recovered_files = Vec::new();

    let image = File::open(path)?;
    let mut reader = BufReader::new(image);
    let mut buffer = vec![0u8; chunk_size];

    let mut last_byte = 0x00;
    let mut current_state = State::SEARCHING;

    let mut file_index = 0;
    let mut output: Option<File> = None;
    let mut current_size = 0usize;

    let mut saw_valid_segment = false;
    let mut current_buffer: Vec<u8> = Vec::new();

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        process_chunk(
            &buffer[..bytes_read],
            last_byte,
            &mut current_state,
            &mut output,
            &mut file_index,
            &mut current_size,
            &mut saw_valid_segment,
            &mut recovered_files,
            &mut current_buffer,
        );

        last_byte = buffer[bytes_read - 1];
    }

    Ok(recovered_files)
}

fn process_chunk(
    chunk: &[u8],
    last_byte: u8,
    current_state: &mut State,
    output: &mut Option<File>,
    file_index: &mut usize,
    current_size: &mut usize,
    saw_valid_segment: &mut bool,
    recovered_files: &mut Vec<String>,
    current_buffer: &mut Vec<u8>,
) {
    let mut skip_next = false;

    if compare_to_marker(
        last_byte,
        chunk[0],
        current_state,
        output,
        file_index,
        current_size,
        saw_valid_segment,
        recovered_files,
        current_buffer,
    ) {
        println!("{:?}, Chunk at boundary", current_state);
    }

    for i in 0..chunk.len().saturating_sub(1) {
        if skip_next {
            skip_next = false;
            continue;
        }

        let x = chunk[i];
        let y = chunk[i + 1];

        let is_marker = compare_to_marker(
            x,
            y,
            current_state,
            output,
            file_index,
            current_size,
            saw_valid_segment,
            recovered_files,
            current_buffer,
        );

        if is_marker {
            skip_next = true;
            continue;
        }

        if *current_state == State::COLLECTING {
            current_buffer.push(x);
            *current_size += 1;

            if check_max_size(*current_size) {
                cleanup_invalid(file_index, output, current_state, current_size, current_buffer);
                return;
            }

            if x == 0xFF && (y == 0xE0 || y == 0xE1) {
                *saw_valid_segment = true;
            }
        }
    }

    if *current_state == State::COLLECTING {
        let last = chunk[chunk.len() - 1];

        current_buffer.push(last);
        *current_size += 1;
    }
}

fn compare_to_marker(
    x: u8,
    y: u8,
    current_state: &mut State,
    output: &mut Option<File>,
    file_index: &mut usize,
    current_size: &mut usize,
    saw_valid_segment: &mut bool,
    recovered_files: &mut Vec<String>,
    current_buffer: &mut Vec<u8>,
) -> bool {
    // START
    if *current_state == State::SEARCHING {
        if x == START_MARKER[0] && y == START_MARKER[1] {

            *current_state = State::COLLECTING;
            println!("JPG found");
            *saw_valid_segment = false;

            *file_index += 1;

            current_buffer.clear();
            current_buffer.push(0xFF);
            current_buffer.push(0xD8);

            *current_size = 2;

            let filename = format!("raw_jpgs/image_{:04}.jpg", file_index);
            let file = File::create(&filename).unwrap();

            *output = Some(file);
            recovered_files.push(filename);

            return true;
        }
    }

    // END
    if *current_state == State::COLLECTING {
        if x == END_MARKER[0] && y == END_MARKER[1] {
            current_buffer.push(0xFF);
            current_buffer.push(0xD9);

            let is_valid = image::load_from_memory(&current_buffer).is_ok();

            let file_path = format!("raw_jpgs/image_{:04}.jpg", file_index);

            if !is_valid || !*saw_valid_segment {
                let _ = std::fs::remove_file(file_path);
                cleanup_invalid(file_index, output, current_state, current_size, current_buffer);
                return true;
            }

            if let Some(file) = output.as_mut() {
                let _ = file.write_all(&current_buffer);
            }

            *output = None;
            *current_state = State::SEARCHING;
            current_buffer.clear();

            return true;
        }
    }

    false
}

fn cleanup_invalid(
    _file_index: &mut usize,
    output: &mut Option<File>,
    state: &mut State,
    size: &mut usize,
    buffer: &mut Vec<u8>,
) {
    *output = None;
    *state = State::SEARCHING;
    *size = 0;
    buffer.clear();
}

fn check_max_size(current_size: usize) -> bool {
    current_size >= MAX_SIZE
}

fn validate_header(_header: &[u8]) -> bool {
    true
}