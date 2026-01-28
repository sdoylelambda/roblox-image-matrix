from pathlib import Path
import struct
import numpy as np
import zlib


def read_png(filename):
    with open(filename, 'rb') as f:
        # Verify the PNG signature
        signature = f.read(8)
        assert signature == b'\x89PNG\r\n\x1a\n'

        chunks = []
        while True:
            # Read the chunk length and type
            length_data = f.read(4)
            chunk_length = struct.unpack('>I', length_data)[0]
            chunk_type = f.read(4)
            chunk_data = f.read(chunk_length)
            crc = f.read(4)  # Read CRC (but we're not going to use it)

            chunks.append((chunk_type, chunk_data))

            if chunk_type == b'IEND':
                break

    return chunks


def parse_ihdr(data):
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack('>IIBBBBB', data)
    return width, height, bit_depth, color_type, compression, filter_method, interlace


def parse_idat(data):
    return zlib.decompress(data)


def image_to_matrix(image_path):
    chunks = read_png(image_path)

    # get IHDR
    ihdr_chunk = next(chunk for chunk in chunks if chunk[0] == b'IHDR')

    # get all IDAT chunks
    idat_chunks = [chunk for chunk in chunks if chunk[0] == b'IDAT']

    # decompress all at once
    raw_data = zlib.decompress(b''.join(chunk[1] for chunk in idat_chunks))

    # Parse IHDR for width and height
    width, height, bit_depth, color_type, compression, filter_method, interlace = parse_ihdr(ihdr_chunk[1])


    # Assuming the image is in RGB format (color_type 2) and 8-bit depth
    expected_size = width * height * 3

    # Convert to a numpy array
    img = np.frombuffer(raw_data, dtype=np.uint8)

    # Adjust the reshape based on the actual data size
    if img.size == expected_size:
        img = img.reshape((height, width, 3))
    else:
        # Calculate the actual height based on the data size
        actual_height = img.size // (width * 3)
        img = img[:actual_height * width * 3].reshape((actual_height, width, 3))

    # Resize the image manually to 34x34
    img_resized = np.zeros((34, 34, 3), dtype=np.uint8)
    x_ratio = width / 34
    y_ratio = height / 34

    for i in range(34):
        for j in range(34):
            px = int(i * y_ratio)
            py = int(j * x_ratio)
            img_resized[i, j] = img[px, py]

    # Initialize a 34x34 matrix to hold the color codes
    color_matrix = []

    for row in img_resized:
        color_row = []
        for pixel in row:
            # Convert each pixel (which is an RGB tuple) to a hex color code
            hex_color = '#{:02x}{:02x}{:02x}'.format(pixel[0], pixel[1], pixel[2])
            color_row.append(hex_color)
        color_matrix.append(color_row)

    return np.array(color_matrix)


def save_matrix_to_file(matrix, output_path):
    with open(output_path, 'w') as f:
        for row in matrix:
            # Join the row elements into a single string separated by spaces
            row_str = ' '.join(row)
            f.write(row_str + '\n')


# Example usage
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Input"

image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

images = [
    p for p in INPUT_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
]

if not images:
    raise RuntimeError("No image found in Input folder")

image_path = images[0]   # ✅ THIS is the file path

print("Using image:", image_path)

output_path = BASE_DIR / "Output/Output.txt"

matrix = image_to_matrix(image_path)
save_matrix_to_file(matrix, output_path)

print(f"Matrix saved to {output_path}")
