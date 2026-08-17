import base64
import os

position1 = f""" 
I will now provide several hand-zoom images with corresponding position labels as the position support set to help you have a sufficient reference when making predictions on test images. 
There are five possible position of hand in cued speech. In this support set, each hand position category contains 8 pictures, which cover all eight possible hand shapes that appear in the same position label. 
Please note the hand position indicates the straight finger are point to.
   
You need to use the following support set as a reference to complete subsequent hand position predictions

Here are the support set:
'hand_position': 0 (Eye)
'hand zoom images':
"""

position1_2 = f"""
Look at the hand zoom images: it is clear that in all images, the straight fingers are pointing to the area around the eye, which corresponds to label 0 (Eye).   
"""

position2 = f"""
'hand_position': 1 (Right side of the head)
'hand zoom images':
"""

position2_2 = f"""
look at the hand zoom images: It is clear that in all images, the straight fingers are not pointing to the facial area and holding at the right side of the head, which corresponds to label 1 (Right side of the head).
"""

position3 = f"""
'hand_position': 2 (Cheek)
'hand zoom images':
"""

position3_2 = f"""
Look at the hand zoom images: it is clear that in all images, the straight fingers are pointing to the area right side of the mouth and around cheek, which corresponds to label 2 (Cheek).
"""

position4 = f"""
'hand_position': 3 (Chin)
'hand zoom images':
"""

position4_2 = f"""
Look at the hand zoom images: it is clear that in all images, the straight fingers are pointing to the chin area where directly below the mouth, corresponds to label 3 (Chin).
"""

position5 = f"""
'hand_position': 4  (Below the head)
'hand zoom images':
 """

position5_2 = f"""
Look at the hand zoom images: It is clear that in all images, the straight fingers do not point to the facial area but to the area below the head, which corresponds to label 4 (below the head).
"""

label_list = [position1, position1_2, position2, position2_2, position3, position3_2, position4, position4_2, position5, position5_2]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def image_content(encoded_image):
    """Build a current Chat Completions image content item."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_image}",
            "detail": "low",
        },
    }


def build_position_support_set(hand_folder_path, set_size=5):
    content_list = []
    for i in range(5):
        index = i+1
        hand_path = os.path.join(hand_folder_path, f"{index + 0*5}.jpg")
        hand_path_2 = os.path.join(hand_folder_path, f"{index + 1*5}.jpg")
        hand_path_3 = os.path.join(hand_folder_path, f"{index + 2*5}.jpg")
        hand_path_4 = os.path.join(hand_folder_path, f"{index + 3*5}.jpg")
        hand_path_5 = os.path.join(hand_folder_path, f"{index + 4*5}.jpg")
        hand_path_6 = os.path.join(hand_folder_path, f"{index + 5*5}.jpg")
        hand_path_7 = os.path.join(hand_folder_path, f"{index + 6*5}.jpg")
        hand_path_8 = os.path.join(hand_folder_path, f"{index + 7*5}.jpg")



        hand_zoom_img = encode_image(hand_path)
        hand_zoom_img_2 = encode_image(hand_path_2)
        hand_zoom_img_3 = encode_image(hand_path_3)
        hand_zoom_img_4 = encode_image(hand_path_4)
        hand_zoom_img_5 = encode_image(hand_path_5)
        hand_zoom_img_6 = encode_image(hand_path_6)
        hand_zoom_img_7 = encode_image(hand_path_7)
        hand_zoom_img_8 = encode_image(hand_path_8)


        """
        sample_text = {
            "type": "text",
            "text": label_list[i],
        }
        sample_image = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{current_img}"},
            "detail": "low"
        }
        """

        content_list.append({"type": "text", "text": label_list[i * 2]})
        content_list.append(image_content(hand_zoom_img))
        content_list.append(image_content(hand_zoom_img_2))
        content_list.append(image_content(hand_zoom_img_3))
        content_list.append(image_content(hand_zoom_img_4))
        content_list.append(image_content(hand_zoom_img_5))
        content_list.append(image_content(hand_zoom_img_6))
        content_list.append(image_content(hand_zoom_img_7))
        content_list.append(image_content(hand_zoom_img_8))
        content_list.append({"type": "text", "text": label_list[i * 2 + 1]})
    return content_list


