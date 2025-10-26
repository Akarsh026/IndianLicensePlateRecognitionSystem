import cv2
import easyocr
import numpy as np
import re
import os

# Mapping of Indian state codes to state names
STATE_CODES = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "WB": "West Bengal",
    "AN": "Andaman and Nicobar Islands",
    "CH": "Chandigarh",
    "DN": "Dadra and Nagar Haveli and Daman & Diu",
    "DL": "Delhi",
    "LD": "Lakshadweep",
    "PY": "Puducherry"
}

class LicensePlateRecognizer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.reader = easyocr.Reader(['en'])

        # Load Haar cascade using relative path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(script_dir, "haarcascade_plate_number.xml")
        self.plate_cascade = cv2.CascadeClassifier(cascade_path)

        if self.plate_cascade.empty():
            raise IOError(f"Error loading cascade classifier from {cascade_path}")

        # Basic OCR corrections
        self.basic_corrections = {
            'N': 'M',
            'Z': '2',
            'O': '0',
            'I': '1',
            'S': '5',
            'B': '8'
        }

    def correct_plate_contextual(self, text):
        """Context-aware correction for Indian plates"""
        corrected = ''
        for i, c in enumerate(text.upper()):
            if i < 4:  # first few chars usually letters
                if c == '0':
                    c = 'D'
                elif c in self.basic_corrections:
                    c = self.basic_corrections[c]
            else:
                if c in self.basic_corrections:
                    c = self.basic_corrections[c]
            corrected += c
        return corrected

    def preprocess_plate(self, plate_roi):
        """Preprocess plate region for OCR"""
        plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.bilateralFilter(plate_gray, 9, 75, 75)  # denoise
        plate_gray = cv2.equalizeHist(plate_gray)  # improve contrast
        return plate_gray

    def detect_plates(self):
        """Detect plates in the image"""
        plates = self.plate_cascade.detectMultiScale(
            self.gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(25, 25)
        )
        return plates

    def recognize_plate(self, plate_roi):
        """Run OCR and apply post-processing"""
        plate_gray = self.preprocess_plate(plate_roi)
        result = self.reader.readtext(plate_gray)

        candidates = []
        for (bbox, text, prob) in result:
            clean_text = ''.join(re.findall(r'[A-Z0-9]', text.upper()))
            if len(clean_text) >= 4:
                candidates.append(clean_text)

        if candidates:
            plate_text = max(candidates, key=len)
            plate_text = self.correct_plate_contextual(plate_text)
            return plate_text
        return None

    def parse_plate_details(self, plate_text):
        """
        Parse Indian plate into state, RTO code, series, and number
        Example: MH12AB1234
        """
        details = {
            "state": None,
            "rto": None,
            "series": None,
            "number": None
        }

        if len(plate_text) >= 10:
            state_code = plate_text[:2]
            rto_code = plate_text[2:4]
            series = plate_text[4:6]
            number = plate_text[6:]

            details["state"] = STATE_CODES.get(state_code, state_code)
            details["rto"] = rto_code
            details["series"] = series
            details["number"] = number

        return details

    def process_image(self, display=True):
        """Detect and recognize plates, annotate image"""
        plates = self.detect_plates()
        if len(plates) == 0:
            print("No plates detected.")
            return []

        detected_plates = []

        for (x, y, w, h) in plates:
            pad = 2
            x1, y1 = max(x - pad, 0), max(y - pad, 0)
            x2, y2 = min(x + w + pad, self.img.shape[1]), min(y + h + pad, self.img.shape[0])
            plate_roi = self.img[y1:y2, x1:x2]

            # Draw rectangle
            cv2.rectangle(self.img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            plate_text = self.recognize_plate(plate_roi)
            if plate_text:
                details = self.parse_plate_details(plate_text)
                detected_plates.append({"plate": plate_text, **details})

                # Annotate image
                cv2.putText(
                    self.img,
                    plate_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                print(f"Detected Plate: {plate_text}")
                print(f"Details: {details}")
            else:
                print("No valid plate text detected in region.")

        if display:
            cv2.imshow("Detected Plates", self.img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detected_plates

if __name__ == "__main__":
    recognizer = LicensePlateRecognizer(os.path.join("Data", "image.jpeg"))
    plates = recognizer.process_image()
    print("All detected plates:", plates)
