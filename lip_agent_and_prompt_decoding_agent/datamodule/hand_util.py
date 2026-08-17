import json

import numpy as np
import torch


hashmap = { "<blank>" : 0,
            "<unk>":1,"b": 2, "p": 3, "m": 4, "f": 5, "d": 6, "t": 7, "n": 8, "l": 9, "g": 10, "k": 11, "h": 12, "j": 13,
            "q": 14, "x": 15, "zh": 16, "ch": 17, "sh": 18, "r": 19, "z": 20, "c": 21, "s": 22, "y": 23, "w": 24,
            "yu": 25, "a": 26, "o": 27, "e": 28, "i": 29, "u": 30, "v": 31, "ai": 32, "ei": 33, "ao": 34, "ou": 35, "er": 36,
            "an": 37, "en": 38, "ang": 39, "eng": 40, "ong": 41, "-": 42}

def label2phone(hand_position, hand_gesture):
    if hand_position == 0:
        vowel_options = ['an', 'e', 'o']
    elif hand_position == 1:
        vowel_options = ['a', 'ou', 'er', 'en']
    elif hand_position == 2:
        vowel_options = ['i', 'v', 'ang']
    elif hand_position == 3:
        vowel_options = ['ai', 'u', 'ao']
    elif hand_position == 4:
        vowel_options = ['eng', 'ong', 'ei']
    else:
        vowel_options = ['<blank>']

    if hand_gesture == 0:
        consonant_options = ['p', 'd', 'zh']
    elif hand_gesture == 1:
        consonant_options = ['k', 'q', 'z']
    elif hand_gesture == 2:
        consonant_options = ['s', 'r', 'h']
    elif hand_gesture == 3:
        consonant_options = ['b', 'n', 'yu']
    elif hand_gesture == 4:
        consonant_options = ['m', 't', 'f']
    elif hand_gesture == 5:
        consonant_options = ['l', 'x', 'w']
    elif hand_gesture == 6:
        consonant_options = ['g', 'j', 'ch']
    elif hand_gesture == 7:
        consonant_options = ['y', 'c', 'sh']
    else:
        consonant_options = ['<blank>']

    return vowel_options, consonant_options


def load_npy(npy_path):
    npy_file = np.load(npy_path)
    # print(hand_position.shape)
    return npy_file

def get_lip_frame_range(slow_groups, frame_num):
    range_list = []
    for i in range(len(slow_groups) - 1):
        range_list.append([slow_groups[i][0], slow_groups[i + 1][0]])

    range_list.append([slow_groups[-1][0], frame_num - 1])
    return range_list


def group_elements2(lst):
    if not lst:
        return []

    result = []
    current_group = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] - lst[i - 1] <= 2:
            current_group.append(lst[i])
        else:
            result.append(current_group[:])
            # result.append(current_group)
            current_group = [lst[i]]

    if current_group not in result:
        result.append(current_group[:])
    return result


def screen_slow_motion_group(hand_position):
    slow_index = []
    for i in range(1, len(hand_position)):
        # print(type(hand_position[i]))
        if np.linalg.norm(hand_position[i] - hand_position[i - 1]) <= 6:
            slow_index.append(i)
    lip_keyframes_list = group_elements2(slow_index)
    return lip_keyframes_list



def get_keyframe_groups(position_path):
    hand_position = load_npy(position_path)
    # print(len(hand_position))
    slow_frame_groups = screen_slow_motion_group(hand_position)
    return slow_frame_groups


def load_hand_recog(hand_recog_path, hand_position_path, frame_num):
    hand_matrix = torch.zeros(frame_num, 44)


    with open(hand_recog_path, 'r') as f:
        hand_data = json.load(f)
    f.close()
    key_frames = hand_data['frame_index']
    hand_results = hand_data['recog_results']


    slow_groups = get_keyframe_groups(hand_position_path)
    """
    range_list = get_lip_frame_range(slow_groups, frame_num)
    
    for i in range(len(key_frames)):
        hand_position = hand_results[i]['hand_position']
        hand_gesture = hand_results[i].get('hand_shape', hand_results[i].get('hand_gesture'))
        if hand_gesture is None:
            raise KeyError("hand result must contain hand_shape or hand_gesture")
        vowel_options, consonant_options = label2phone(hand_position, hand_gesture)
        for vowel in vowel_options:
            vowel_index = hashmap[vowel]
            for j in range(range_list[i][0], range_list[i][1]):
                hand_matrix[j][vowel_index] = 1

        for consonant in consonant_options:
            consonant_index = hashmap[consonant]
            for j in range(range_list[i][0], range_list[i][1]):
                hand_matrix[j][consonant_index] = 1
    """
    for i in range(len(key_frames)):
        hand_position = hand_results[i]['hand_position']
        hand_gesture = hand_results[i].get('hand_shape', hand_results[i].get('hand_gesture'))
        if hand_gesture is None:
            raise KeyError("hand result must contain hand_shape or hand_gesture")
        vowel_options, consonant_options = label2phone(hand_position, hand_gesture)

        for vowel in vowel_options:
            vowel_index = hashmap[vowel]
            for j in range(slow_groups[i][0], slow_groups[i][-1]+1):
                hand_matrix[j][vowel_index] = 1

        for consonant in consonant_options:
            consonant_index = hashmap[consonant]
            for j in range(slow_groups[i][0], slow_groups[i][-1]+1):
                hand_matrix[j][consonant_index] = 1

    return hand_matrix


