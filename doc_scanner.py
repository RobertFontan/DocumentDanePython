import cv2
import numpy as np

def capture_image():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    print("Press 's' to capture image, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the live video feed
        cv2.imshow('Webcam', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 's' to capture
            cv2.imwrite('captured_image.jpg', frame)
            print("Image captured and saved.")
            break
        elif key == ord('q'):  # 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply MedianBlur to reduce noise and preserve edges (better than Gaussian for glare reduction)
    blurred = cv2.medianBlur(gray, 5)
    
    # Adjust contrast and brightness to improve edge detection
    adjusted = cv2.convertScaleAbs(blurred, alpha=1.5, beta=30)
    
    # Detect edges using Canny edge detection (adjust thresholds)
    edged = cv2.Canny(adjusted, 30, 120)
    
    return edged

def find_contours(edged_image):
    # Find contours in the edged image
    contours, _ = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, keeping only large ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If the approximated contour has 4 points, assume it's the document
        if len(approx) == 4:
            return approx

    return None

def transform_perspective(image, contour):
    # Get a consistent top-down view of the document
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # Determine the order of the points: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Compute the width and height of the new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the top-down view of the document
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Apply the perspective transformation matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def enhance_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to make the document more readable
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return enhanced

def main():
    # Step 1: Capture the image from webcam
    image = capture_image()

    if image is None:
        return

    # Step 2: Preprocess the image (convert to grayscale, detect edges)
    edged_image = preprocess_image(image)
    
    # Debug: Show edged image
    cv2.imshow('Edged Image', edged_image)
    cv2.waitKey(0)

    # Step 3: Find the contours (document boundaries)
    contour = find_contours(edged_image)

    if contour is None:
        print("No document found.")
        return

    # Step 4: Apply perspective transformation
    warped_image = transform_perspective(image, contour)

    # Step 5: Enhance the warped image
    enhanced_image = enhance_image(warped_image)

    # Step 6: Display and save the final scanned document
    cv2.imshow('Scanned Document', enhanced_image)
    cv2.imwrite('scanned_document.jpg', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
