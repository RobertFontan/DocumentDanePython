import cv2
import numpy as np
import os

def load_image(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None
    
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error: Could not read image file '{file_path}'.")
        return None
    
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_rectangles(image):
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to enhance rectangles
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Check for rectangular shape (4 vertices)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            aspect_ratio = w / float(h)
            
            # Adjust these values to include smaller rectangles
            if 50 < area < image.shape[0] * image.shape[1] * 0.25 and 0.1 < aspect_ratio < 10:
                rectangles.append((x, y, w, h))
    
    return rectangles

def draw_rectangles(image, rectangles):
    for (x, y, w, h) in rectangles:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

def main():
    file_path = './form_templates//image.png'
    image = load_image(file_path)

    if image is None:
        return

    preprocessed = preprocess_image(image)
    rectangles = detect_rectangles(preprocessed)
    
    result_image = image.copy()
    draw_rectangles(result_image, rectangles)
    
    cv2.imshow('Detected Rectangles', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()