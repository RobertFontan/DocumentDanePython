import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    orig = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 75, 200)

    return orig, edges


def find_document_contours(edges):
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area, keep only the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # If the contour has 4 points, assume it's the document
        if len(approx) == 4:
            return approx

    return None


def transform_document(image, contour):
    # Obtain a consistent order of the document points
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # Find top-left, top-right, bottom-right, and bottom-left points
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Get the top-down view of the document
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the transform
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

    # Perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def extract_text_from_document(image):
    # Configure the Tesseract executable path (Windows only)
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Convert image to grayscale (if needed)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply OCR
    text = pytesseract.image_to_string(gray)

    return text


def scan_document(image_path):
    # Preprocess image
    orig, edges = preprocess_image(image_path)

    # Find document contours
    document_contour = find_document_contours(edges)

    if document_contour is None:
        print("Document contour not found.")
        return None

    # Transform the document to get a top-down view
    warped = transform_document(orig, document_contour)

    # Save or display the warped document (optional)
    cv2.imshow("Scanned Document", warped)
    cv2.waitKey(0)

    # Extract text using Tesseract
    text = extract_text_from_document(warped)
    
    print('made it here') 

# Usage
image_path = './form_templates/formi9.jpg'
text_from_document = scan_document(image_path)
print(text_from_document)
