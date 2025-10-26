# Indian License Plate Recognition System

A Python-based license plate detection and recognition system specifically designed for Indian vehicle number plates. The system uses computer vision and OCR techniques to detect, extract, and parse Indian license plates from images.

## Features

- **Automatic Plate Detection**: Uses Haar Cascade classifier to detect license plates in images
- **OCR Recognition**: Leverages EasyOCR for text extraction from detected plates
- **Context-Aware Correction**: Applies intelligent corrections based on Indian plate format
- **State Recognition**: Automatically identifies the state from the plate code
- **Detailed Parsing**: Extracts state, RTO code, series, and number from the plate
- **Visual Output**: Annotates detected plates on the original image

## Requirements

```bash
opencv-python
easyocr
numpy
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install opencv-python easyocr numpy
```

3. Ensure you have the Haar Cascade file:
   - Place `haarcascade_plate_number.xml` in the same directory as the script

## Usage

### Basic Usage

```python
from license_plate_recognizer import LicensePlateRecognizer

# Initialize recognizer with image path
recognizer = LicensePlateRecognizer("path/to/image.jpg")

# Process image and display results
plates = recognizer.process_image(display=True)

# Print all detected plates
print("All detected plates:", plates)
```

### Command Line

```bash
python license_plate_recognizer.py
```

Note: Update the image path in the `__main__` section before running.

## Indian License Plate Format

The system is designed to recognize Indian license plates which follow this format:
- **State Code** (2 letters): e.g., MH, KA, DL
- **RTO Code** (2 digits): e.g., 12, 01, 09
- **Series** (2 letters): e.g., AB, CD, XY
- **Number** (4 digits): e.g., 1234, 5678

Example: `MH12AB1234`

## Supported States

The system recognizes all Indian states and union territories:
- 28 States (e.g., Maharashtra-MH, Karnataka-KA, Delhi-DL)
- 8 Union Territories (e.g., Chandigarh-CH, Puducherry-PY)

## How It Works

1. **Image Preprocessing**: Converts image to grayscale for detection
2. **Plate Detection**: Uses Haar Cascade to locate potential license plates
3. **ROI Extraction**: Extracts regions of interest with padding
4. **Image Enhancement**: Applies bilateral filtering and histogram equalization
5. **OCR Recognition**: Uses EasyOCR to extract text
6. **Post-Processing**: Applies context-aware corrections
7. **Parsing**: Breaks down plate into state, RTO, series, and number

## Output Format

The system returns a list of detected plates with the following structure:

```python
{
    "plate": "MH12AB1234",
    "state": "Maharashtra",
    "rto": "12",
    "series": "AB",
    "number": "1234"
}
```

## File Structure

```
├── license_plate_recognizer.py
├── haarcascade_plate_number.xml
├── Data/
│   └── image.jpeg
└── README.md
```

## Key Classes and Methods

### `LicensePlateRecognizer`

- `__init__(image_path)`: Initialize with image path
- `detect_plates()`: Detect license plates in the image
- `recognize_plate(plate_roi)`: Extract text from plate region
- `parse_plate_details(plate_text)`: Parse plate components
- `process_image(display=True)`: Main processing pipeline

## OCR Corrections

The system includes intelligent character corrections:
- Context-aware: Different corrections for letter vs number positions
- Common substitutions: 0→D, O→0, I→1, S→5, B→8

## Limitations

- Requires good image quality and proper lighting
- Works best with standard Indian license plate formats
- May struggle with heavily damaged or obscured plates
- Depends on Haar Cascade accuracy for detection

## Troubleshooting

**No plates detected:**
- Ensure image has sufficient resolution
- Check lighting conditions
- Verify plate is clearly visible

**Incorrect OCR results:**
- Image quality may be poor
- Adjust preprocessing parameters
- Check if plate follows standard format

**Cascade file error:**
- Ensure `haarcascade_plate_number.xml` is in the correct location
- Verify file is not corrupted

## Future Improvements

- Support for new BH (Bharat) series plates
- Deep learning-based detection (YOLO/SSD)
- Multi-line plate support
- Real-time video processing
- Enhanced preprocessing for difficult conditions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Akarsh 