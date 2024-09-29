import cv2
import pytesseract
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture(0)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Create a named window
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 640, 480)

while True:
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from camera")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Draw rectangle on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Extract ROI
            roi = gray[y:y+h, x:x+w]

            # Perform OCR on ROI
            text = pytesseract.image_to_string(roi)
            print("Detected text:", text)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()