import os 

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./iconic-reactor-437022-u3-bf365d854132.json"

def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    
    print(client)
    
    

# Open a file in write mode
    with open('extracted_texts.txt', 'w') as file:
        texts = response.text_annotations
        file.write("Texts:\n")
    
        for text in texts:
            file.write(f'\n"{text.description}"\n')
    
            vertices = [
                f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices
            ]
    
            file.write("bounds: {}\n".format(",".join(vertices)))
    
        if response.error.message:
            raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )


def detect_labels(path):
    """Detects labels in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations
    print("Labels:")

    for label in labels:
        print(label.description)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
        
        

import cv2 


# i want the bounds to be a list of tuples

bounds1 = [(190,33),(259,32),(259,45),(190,46)]


def draw_bounding_box(image ,bounds ):
    
    
#"ensure"
#bounds: (153,88),(178,88),(178,97),(153,97)
    
   # Extract coordinates from bounds
    x1, y1 = bounds[0]
    x2, y2 = bounds[2]  # Using the diagonal point for bottom-right corner

    # Calculate width and height
    w = x2 - x1
    h = y2 - y1

    # Draw the rectangle
    color = (0, 255, 0)  # Green color
    thickness = 2
    img_with_box = cv2.rectangle(image, (x1, y1), (x1+w, y1+h), color, thickness)

    # Optionally, add the word as text above the rectangle

    return img_with_box

# while there are still rectangles to process
#    draw the next one on



#cv2.imshow('Box Test', draw_bounding_box(cv2.imread('./form_templates/image.png'), bounds1))
cv2.waitKey(0)

detect_text('./form_templates/image.png')
#detect_labels('./form_templates/image.png')