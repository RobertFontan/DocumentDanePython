import cv2
from google.cloud import vision
import numpy as np

def get_text_width(text, font, font_scale, thickness):
    """Helper function to calculate the width of the text when rendered."""
    size = cv2.getTextSize(text, font, font_scale, thickness)
    return size[0][0]  # Return the width of the text

def detect_blocks_side_by_side(path):
    """Detects text blocks in the file, draws bounding boxes, and displays text blocks in a separate dynamic window."""
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)  # Using document_text_detection for block-level detection
    
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    # Read the image using OpenCV
    img = cv2.imread(path)

    # Initialize a list to store block summaries
    summaries = []

    # Iterate through detected pages
    for page in response.full_text_annotation.pages:
        # Iterate through blocks within the page
        for block in page.blocks:
            # Get block vertices (bounding box coordinates)
            vertices = [(vertex.x, vertex.y) for vertex in block.bounding_box.vertices]
            pts = np.array(vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Draw bounding box for each block
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Extract block text using your logic
            block_text = ' '.join([
                ''.join([symbol.text for symbol in word.symbols]) for paragraph in block.paragraphs for word in paragraph.words
            ])
            summaries.append(block_text)

    # Create a dynamically sized window to display the text annotations
    text_height = 25  # Height for each line of text
    padding = 20      # Padding between the text and the edges
    num_text_blocks = len(summaries)  # Number of detected blocks
    text_window_height = num_text_blocks * text_height + 2 * padding

    # Get the height of the original document image
    doc_height, doc_width, _ = img.shape

    # If the text window is shorter than the document, adjust its height
    text_window_height = max(text_window_height, doc_height)

    # Determine the required width for the text window based on the longest block text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    max_text_width = 0
    for block_text in summaries:
        text_width = get_text_width(f"Block: {block_text}", font, font_scale, thickness)
        max_text_width = max(max_text_width, text_width)

    # Add some padding to the maximum width to prevent text from hitting the edge
    text_window_width = max_text_width + 30

    # Create a blank white image for the text window with the dynamically calculated width
    text_img = np.ones((text_window_height, int(text_window_width), 3), dtype=np.uint8) * 255  # White background

    # Add the detected block text to the text window
    y_offset = padding  # Start from the top padding
    for i, block_text in enumerate(summaries):
        # Write the block text in the text window
        cv2.putText(text_img, f"Block {i + 1}: {block_text}", (10, y_offset), 
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y_offset += text_height  # Move to the next line for the next block

    # Ensure both images (img and text_img) have the same height
    if text_img.shape[0] < img.shape[0]:
        # If the text window is shorter, pad the text image
        padding = np.ones((img.shape[0] - text_img.shape[0], text_img.shape[1], 3), dtype=np.uint8) * 255
        text_img = np.vstack((text_img, padding))
    elif img.shape[0] < text_img.shape[0]:
        # If the document image is shorter, pad the document image
        padding = np.ones((text_img.shape[0] - img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
        img = np.vstack((img, padding))

    # Combine the two images (document + text window) side by side
    combined_img = np.hstack((img, text_img))

    # Save and display the combined image
    cv2.imwrite("images/combined_output.jpg", combined_img)

    while True:
        # Display the image with bounding boxes and text window
        cv2.imshow('Document with Block Summaries on the Side', combined_img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    detect_blocks_side_by_side("form_templates/formi9.jpg")
