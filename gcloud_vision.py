import cv2
from google.cloud import vision
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import numpy as np

# Initialize Vertex AI and the Generative Model (Gemini)
project_id = "skilled-index-437101-j1"
location = "us-central1"
vertexai.init(project=project_id, location=location)

# Load the Gemini model
model = GenerativeModel(model_name="gemini-1.0-pro")

def get_text_width(text, font, font_scale, thickness):
    """Helper function to calculate the width of the text when rendered."""
    size = cv2.getTextSize(text, font, font_scale, thickness)
    return size[0][0]  # Return the width of the text

def summarize_text_with_gemini(text):
    """Uses Gemini model (Vertex AI) to summarize text."""
    # Start a chat session
    chat = model.start_chat(response_validation=False)

    # Custom instruction/prompt for the Gemini model to summarize the text
    prompt = f"Summarize the following text in one sentence, taking into consideration a user trying to fill out the form boxes in an insurance document: {text}"

    # Generate the summary
    response = chat.send_message(prompt)

    summarized_text = response.text.strip()  # Get and clean the response text
    print(f"Summarized Text: {summarized_text}")  # Log the summary
    return summarized_text

def detect_and_summarize_from_image(path):
    """Detects text blocks from an image file using Google Cloud Vision API, sends each block to Vertex AI for summarization, and displays the result."""
    vision_client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Using document_text_detection to extract text from image
    response = vision_client.document_text_detection(image=image)
    
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    # Read the image using OpenCV
    img = cv2.imread(path)

    # Initialize a list to store Vertex AI summaries
    summaries = []

    # Font settings for block numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 0, 0)  # Blue color for block numbers

    # Block numbering starts from 1
    block_number = 1

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

            # Extract block text using Vision API
            block_text = ' '.join([
                ''.join([symbol.text for symbol in word.symbols]) for paragraph in block.paragraphs for word in paragraph.words
            ])
            
            print(f"Extracted Block Text: {block_text}")  # Log the block text

            # Send the extracted text to Gemini (Vertex AI) for summarization
            block_summary = summarize_text_with_gemini(block_text)
            summaries.append(block_summary)

            # Draw the block number in the top-left corner of the bounding box
            # Calculate the position to draw the number (top-left corner of the block)
            text_position = (vertices[0][0] + 5, vertices[0][1] - 5)  # Adjust offset
            cv2.putText(img, f'{block_number}', text_position, font, font_scale, color, thickness, cv2.LINE_AA)

            # Increment block number for the next block
            block_number += 1

    # Create a dynamically sized window to display the text summaries
    text_height = 25  # Height for each line of text
    padding = 20      # Padding between the text and the edges
    num_text_blocks = len(summaries)  # Number of detected blocks
    text_window_height = num_text_blocks * text_height + 2 * padding

    # Get the height of the original document image
    doc_height, doc_width, _ = img.shape

    # If the text window is shorter than the document, adjust its height
    text_window_height = max(text_window_height, doc_height)

    # Determine the required width for the text window based on the longest summary
    max_text_width = 0
    for block_summary in summaries:
        text_width = get_text_width(f"Summary: {block_summary}", font, font_scale, thickness)
        max_text_width = max(max_text_width, text_width)

    # Add some padding to the maximum width to prevent text from hitting the edge
    text_window_width = max_text_width + 30

    # Create a blank white image for the text window with the dynamically calculated width
    text_img = np.ones((text_window_height, int(text_window_width), 3), dtype=np.uint8) * 255  # White background

    # Add the Vertex AI-generated summaries to the text window
    y_offset = padding  # Start from the top padding
    for i, block_summary in enumerate(summaries):
        # Write the block summary in the text window
        cv2.putText(text_img, f"Block {i + 1} Summary: {block_summary}", (10, y_offset), 
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
    cv2.imwrite("images/combined_output_vertex_ai.jpg", combined_img)

    while True:
        # Display the image with bounding boxes and text window
        cv2.imshow('Document with Vertex AI Summaries on the Side', combined_img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    detect_and_summarize_from_image("form_templates/formi9.jpg")
