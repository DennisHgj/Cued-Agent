import base64
import os

shape0 = f""" 
   I will now provide several hand-zoom images with corresponding labels as the shape support set to help you have a sufficient reference when making predictions on test images. 
   There are eight types of hand shapes. Five images for hand at different locations with descriptions are provided for each category. Please ignore the hand's color and position and focus only on the hand shape.
   
    You need to use the following support set as a reference to complete subsequent predictions

   Here are the support set:
   'hand_shape': 0 (Index finger straight only. Thumb, middle, ring and pinky finger are bent.)
   'hand zoom images':
   """

shape0_2 = f"""
    Look at the hand zoom images:
    it is clear that only the index finger is straight in each image, which corresponds to label 0 (Index finger straight only. Thumb, middle, ring and pinky finger are bent and not visible.)
    """

shape1 = f"""
   'hand_shape': 1 (Index and middle fingers are straight and Together. Thumb, ring and pinky fingers are bent.)
   'hand zoom images':
"""

shape1_2 = f"""
    look at the hand zoom images: it is clear that only index and middle fingers are straight and close together in each image, which corresponds to label 1 (Index and middle fingers are straight and Together. Thumb, ring and pinky fingers are bent.)
    Also the shape label 1 is easily confused with label 7 (Index and middle fingers are straight and Apart. Thumb, ring and pinky fingers are bent.) and label 6 (Thumb, index and middle fingers are straight. Ring and pinky fingers are bent.)
    The obvious difference between shape 1 and 7 is that in shape 1, the index and middle fingers are close together.
    Then the obvious difference between shape 1 and 6 is that in shape 1, the thumb is bent.
"""

shape2 = f"""
   'hand_shape': 2 (Middle, ring and pinky fingers are straight. Thumb and index fingers are bent.)
    'hand zoom images':
"""

shape2_2 = f"""
    Look at the hand zoom images: it is clear that only middle, ring and pinky fingers are straight in each image, which corresponds to label 2 (Middle, ring and pinky fingers are straight. Thumb and index fingers are bent.)
    Also the shape label 2 is easily confused with label 3 (Index, middle, ring and pinky fingers are straight. Thumb is bent.)
    The obvious difference between them is that in shape 2, the index finger is bent，only three fingers are straight. 
"""

shape3 = f"""
    'hand_shape': 3 (Index, middle, ring and pinky fingers are straight. Thumb is bent.)
    'hand zoom images':
"""

shape3_2 = f"""
    Look at the hand zoom images: it is clear that only thumb is bent and not visible in each image, which corresponds to label 3 (Index, middle, ring and pinky fingers are straight. Thumb is bent.)
    Also the shape label 3 is easily confused with label 2 (Middle, ring and pinky fingers are straight. Thumb and index fingers are bent.) and label 4 (All fingers straight).
    The obvious difference between shape 3 and shape 2 is that in shape 3, the index finger is straight and visible，four fingers are straight.
    Then the obvious difference between shape 3 and shape 4 is that in shape 3, the thumb is bent and not visible.
"""

shape4 = f"""
    'hand_shape': 4 (All fingers straight)
    'hand zoom images':
   """

shape4_2 = f"""
    Look at the hand zoom images: it is clear that all five fingers are straight in each image, which corresponds to label 4 (All fingers straight.)
    Also the shape label 4 is easily confused with label 3 (Index, middle, ring and pinky fingers are straight. Thumb is bent.)
    The obvious difference between them is that in shape 4, thumb is straight and visible.
    """

shape5 = f"""
    'hand_shape': 5 (Thumb and index fingers are straight. Middle, ring and pinky fingers are bent.)
    'hand zoom images':
    """

shape5_2 = f"""
    Look at the hand zoom images: it is clear that only thumb and index fingers are straight in each image, which corresponds to label 5 (Thumb and index fingers are straight. Middle, ring and pinky fingers are bent.)
    Also the shape label 5 is easily confused with label 6 (Thumb, index and middle fingers are straight. Ring and pinky fingers are bent.)
    The obvious difference between them is that in shape 5, the middle finger is bent and not visible.
"""

shape6= f"""
   'hand_shape': 6 (Thumb, index and middle fingers are straight. Ring and pinky fingers are bent.)
   'hand zoom images':
   
    """

shape6_2 = f"""
    Look at the hand zoom images: it is clear that thumb, index and middle fingers are straight in each image, which corresponds to label 6 (Thumb, index and middle fingers are straight. Ring and pinky fingers are bent.)
    Also the shape label 6 is easily confused with label 5 (Thumb and index fingers are straight. Middle, ring and pinky fingers are bent.) and label 1 (Index and middle fingers are straight and Together. Thumb, ring and pinky fingers are bent.)
    The obvious difference between label 5 and 6 is that in shape 6, the middle finger is straight and visible.
    Then the obvious difference between shape 6 and 1 is that in shape 6, the thumb is straight.
"""

shape7 = """
    'hand_shape': 7 (Index and middle fingers are straight and Apart. Thumb, ring and pinky fingers are bent.)
    'hand zoom images':
   """

shape7_2 = f"""
    Look at the hand zoom images: it is clear that the index and middle fingers are straight and not close together in each image, which corresponds to label 7 (Index and middle fingers are straight and Apart. Thumb, ring and pinky fingers are bent.)
    Also the shape label 7 is easily confused with label 1 (Index and middle fingers are straight and Together. Thumb, ring and pinky fingers are bent and not visible).
    The obvious difference between them is that in shape 7, the index and middle fingers are not close together. 
"""



label_list = [shape0, shape0_2, shape1, shape1_2, shape2, shape2_2, shape3, shape3_2, shape4, shape4_2,
              shape5, shape5_2, shape6, shape6_2, shape7, shape7_2]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def image_content(encoded_image, detail="high"):
    """Build a current Chat Completions image content item."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_image}",
            "detail": detail,
        },
    }


def build_shape_support_set(hand_folder_path, set_size=5, detail="high"):
    content_list = []
    for i in range(8):
        index = i * set_size
        hand_path = os.path.join(hand_folder_path, f"{index + 1}.jpg")
        hand_path_2 = os.path.join(hand_folder_path, f"{index + 2}.jpg")
        hand_path_3 = os.path.join(hand_folder_path, f"{index + 3}.jpg")
        hand_path_4 = os.path.join(hand_folder_path, f"{index + 4}.jpg")
        hand_path_5 = os.path.join(hand_folder_path, f"{index + 5}.jpg")


        hand_zoom_img = encode_image(hand_path)
        hand_zoom_img_2 = encode_image(hand_path_2)
        hand_zoom_img_3 = encode_image(hand_path_3)
        hand_zoom_img_4 = encode_image(hand_path_4)
        hand_zoom_img_5 = encode_image(hand_path_5)


        content_list.append({"type": "text", "text": label_list[i * 2]})
        content_list.append(image_content(hand_zoom_img, detail))
        content_list.append(image_content(hand_zoom_img_2, detail))
        content_list.append(image_content(hand_zoom_img_3, detail))
        content_list.append(image_content(hand_zoom_img_4, detail))
        content_list.append(image_content(hand_zoom_img_5, detail))
        content_list.append({"type": "text", "text": label_list[i * 2 + 1]})
    return content_list
