import cv2
import pytesseract

# Set the tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def highlight_text_in_image():
    # Initialize the webcam
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
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use pytesseract to detect text and get bounding boxes
        boxes = pytesseract.image_to_boxes(gray)
        
        # Draw rectangles around detected text
        for b in boxes.splitlines():
            b = b.split(' ')
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(frame, (x, frame.shape[0] - y), (w, frame.shape[0] - h), (0, 255, 0), 2)
        
        # Display the frame with highlighted text
        cv2.imshow('Text Highlight', frame)
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            # Extract text from the image
            extracted_text = pytesseract.image_to_string(gray)
            print("Extracted Text:")
            print(extracted_text)
            
            # Save the captured image
            cv2.imwrite('../images/captured_image.png', frame)
            print("Image captured and saved as 'captured_image.png'.")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    highlight_text_in_image()