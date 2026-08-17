import cv2
import numpy as np
import os
from pathlib import Path
from mediapipe import solutions

# parameter
lip_box = 140
hand_box = 320

# Predictors are initialized lazily so importing the inference package does not
# depend on the caller's current working directory or allocate MediaPipe state.
detector = None
lip_predictor = None
hand_predictor = None
_PROJECT_ROOT = Path(__file__).resolve().parent


def _get_lip_models():
    global detector, lip_predictor
    if detector is None or lip_predictor is None:
        try:
            import dlib
        except ImportError as exc:
            raise RuntimeError(
                "dlib is required only for the legacy dataset segmentation path"
            ) from exc
        predictor_path = _PROJECT_ROOT / "shape_predictor_68_face_landmarks.dat"
        if not predictor_path.is_file():
            raise FileNotFoundError(f"Face landmark predictor not found: {predictor_path}")
        detector = dlib.get_frontal_face_detector()
        lip_predictor = dlib.shape_predictor(str(predictor_path))
    return detector, lip_predictor


def _get_hand_predictor():
    global hand_predictor
    if hand_predictor is None:
        hand_predictor = solutions.hands.Hands()
    return hand_predictor


def get_lip_margin(frame_copy):
    frameGray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    face_detector, predictor = _get_lip_models()
    faces = face_detector(frameGray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        # imgOriginal = cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
        landmarks = predictor(frameGray, face)
        myPoints = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x, y])

        myPoints = np.array(myPoints)
        # 嘴唇区域mask提取
        x, y, w, h = cv2.boundingRect(myPoints[48:61])
        return x, y, w, h


def get_hand_margin(frame_copy):
    frameRGB = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    results = _get_hand_predictor().process(frameRGB)

    if results.multi_hand_landmarks:
        landmark_list = []
        # print(results.multi_hand_landmarks)
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark_id, finger_axis in enumerate(hand_landmarks.landmark):
                # 便利某个手的每个关节
                # print(finger_axis.x)
                h, w, c = frameRGB.shape
                landmark_list.append([finger_axis.x * w, finger_axis.y * h])
        myPoints = np.array(landmark_list)

        x, y, w, h = cv2.boundingRect(myPoints.astype(np.int32))
        return x, y, w, h
    return None


def segment(data_path, person_dirs):
    person_paths = []
    for person in person_dirs:
        # if len(person) == 2 and person=='xp':
        person_paths.append(person)

    for p_dir in person_paths:
        person_path = os.path.join(data_path, p_dir)
        lip_save_path = os.path.join(data_path, 'segmented_0914', p_dir, 'lip')
        hand_save_path = os.path.join(data_path, 'segmented_0914', p_dir, 'hand')
        hand_pos_path = os.path.join(data_path, 'segmented_0914', p_dir, 'position')
        seg_path = os.path.join(data_path, 'segmented_0914', p_dir, 'seg_video')
        index_save_path = os.path.join(data_path, 'segmented_0914', p_dir, 'frame_index')
        if not os.path.isdir(lip_save_path):
            os.makedirs(lip_save_path)
        if not os.path.isdir(hand_save_path):
            os.makedirs(hand_save_path)
        if not os.path.isdir(hand_pos_path):
            os.makedirs(hand_pos_path)
        if not os.path.isdir(seg_path):
            os.makedirs(seg_path)
        if not os.path.isdir(index_save_path):
            os.makedirs(index_save_path)


        all_files = os.listdir(person_path)
        num_files = 0
        for name in all_files:
            ext = os.path.splitext(name)[1]
            pre = os.path.splitext(name)[0]
            idx = int(pre.split('-')[1])

            if ext == '.mp4':
                num_files = num_files + 1
                video_path = person_path + '/' + pre + ".mp4"
                print(video_path)

                videoCapture = cv2.VideoCapture(video_path)

                # video information
                fps = int(round(videoCapture.get(cv2.CAP_PROP_FPS)))
                width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_counter = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
                # print(height)
                # print(width)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                lip_video_path = lip_save_path + '/' + pre + '.mp4'
                hand_video_path = hand_save_path + '/' + pre + '.mp4'
                hand_pos_npy = hand_pos_path + '/' + pre + '.npy'
                seg_video_path = seg_path + '/' + pre + '.mp4'
                frame_idx_path = index_save_path + '/' + pre + '.npy'
                if os.path.exists(lip_video_path) and os.path.exists(hand_video_path) and os.path.exists(hand_pos_npy):
                    #if os.path.getsize(seg_path)/ os.path.getsize(video_path) >= 0.4:
                        #continue
                   # else:
                        #print('re-segmenting:', video_path)
                    print('existing:', video_path)
                    continue

                lip_videoWriter = cv2.VideoWriter(lip_video_path, fourcc, fps, (lip_box, lip_box))
                hand_videoWriter = cv2.VideoWriter(hand_video_path, fourcc, fps, (hand_box, hand_box))
                seg_videoWriter = cv2.VideoWriter(seg_video_path, fourcc, fps, (width, height))
                hand_position = []
                frame_idx_list = []

                frame_idx = 0

                while True:
                    success, frame = videoCapture.read()


                    frame_idx = frame_idx + 1
                    if success:
                        frame_copy = frame.copy()
                        original_frame = frame.copy()
                        lip_point_info = get_lip_margin(frame_copy)
                        hand_point_info = get_hand_margin(frame_copy)

                        if hand_point_info != None and lip_point_info != None:
                            # print(hand_point_info)
                            # print(lip_point_info)

                            # lip segmentation

                            lip_x, lip_y, _, _ = lip_point_info
                            lip_y = lip_y - 40
                            lip_x = lip_x - 22
                            if lip_y < 0:
                                lip_y = 0
                            if lip_x < 0:
                                lip_x = 0

                            if (lip_y + lip_box) >= frame.shape[0]:
                                lip_y = lip_y - ((lip_y + lip_box) - frame.shape[0])
                                if lip_y < 0:
                                    lip_y = 0

                            if (lip_x + lip_box) >= frame.shape[1]:
                                lip_x = lip_x - ((lip_x + lip_box) - frame.shape[1])
                                if lip_x < 0:
                                    lip_x = 0

                            lip_frame = frame[lip_y:lip_y + lip_box, lip_x:lip_x + lip_box, :]

                            # hand segmentation
                            hand_x, hand_y, _, _ = hand_point_info
                            hand_y = hand_y - 40
                            hand_x = hand_x - 40
                            if hand_x < 0:
                                hand_x = 0
                            if hand_y < 0:
                                hand_y = 0
                            frame_copy = frame.copy()
                            if (hand_y + hand_box) > frame.shape[0]:
                                hand_y = hand_y - ((hand_y + hand_box) - frame.shape[0])
                                if hand_y < 0:
                                    hand_y = 0
                            if (hand_x + hand_box) > frame.shape[1]:
                                hand_x = hand_x - ((hand_x + hand_box) - frame.shape[1])
                                if hand_x < 0:
                                    hand_x = 0
                            hand_frame = frame_copy[hand_y:hand_y + hand_box, hand_x:hand_x + hand_box, :]

                            # print(hand_frame.shape)
                            lip_videoWriter.write(lip_frame)
                            hand_videoWriter.write(hand_frame)
                            seg_videoWriter.write(original_frame)
                            hand_position.append([hand_x, hand_y])

                            frame_idx_list.append(frame_idx)

                    else:
                        videoCapture.release()
                        lip_videoWriter.release()
                        hand_videoWriter.release()
                        seg_videoWriter.release()
                        hand_position = np.array(hand_position)
                        np.save(hand_pos_npy, hand_position)
                        frame_idx_list = np.array(frame_idx_list)
                        np.save(frame_idx_path, frame_idx_list)

                        break

                print(video_path + ' is segmented!')


# def test(data_path):
#     person_dirs = os.listdir(data_path)
#     person_paths = []
#     for person in person_dirs:
#         if len(person) == 2:
#             person_paths.append(person)

#     error = 0
#     for p_dir in person_paths:
#         person_path = os.path.join(data_path, p_dir)

#         lip_save_path = os.path.join(data_path, 'segmented', p_dir, 'lip')
#         hand_save_path = os.path.join(data_path, 'segmented', p_dir, 'hand')
#         # print(lip_save_path)


#         all_files = os.listdir(person_path)
#         # print(len(all_files))
#         num_files = 0
#         for name in all_files:
#             ext = os.path.splitext(name)[1]
#             pre = os.path.splitext(name)[0]
#             if ext == '.mp4':
#                 num_files = num_files + 1
#                 lip_video_path = lip_save_path + '/' + pre + ".mp4"
#                 lip_videoCapture = cv2.VideoCapture(lip_video_path)
#                 hand_video_path = lip_save_path + '/' + pre + ".mp4"
#                 hand_videoCapture = cv2.VideoCapture(hand_video_path)

#                 # video information
#                 lip_frame_counter = int(lip_videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
#                 hand_frame_counter = int(lip_videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

#                 print('lip_frame_counter',hand_frame_counter)
#                 print('hand_frame_counter',hand_frame_counter)
#                 if lip_frame_counter!=hand_frame_counter:
#                     error+=1
#                 else:
#                     pass


#     print(error)

def single_video_segment(video_path):
    """
    对单个视频进行分割,返回唇部、手部帧列表和位置信息

    Args:
        video_path: 视频文件路径

    Returns:
        lip_frames: 唇部帧列表
        hand_frames: 手部帧列表
        hand_positions: 手部位置列表 [(x, y), ...]
        frame_indices: 有效帧的索引列表
    """
    videoCapture = cv2.VideoCapture(video_path)

    lip_frames = []
    hand_frames = []
    hand_positions = []
    frame_indices = []

    # Original-video indices are zero based. The previous implementation used
    # one-based indices, which shifted every hand prompt by one frame.
    frame_idx = -1

    while True:
        success, frame = videoCapture.read()
        frame_idx += 1

        if success:
            frame_copy = frame.copy()
            #lip_point_info = get_lip_margin(frame_copy)
            hand_point_info = get_hand_margin(frame_copy)

            if hand_point_info is not None:
                """
                # lip segmentation
                lip_x, lip_y, _, _ = lip_point_info
                lip_y = lip_y - 40
                lip_x = lip_x - 22
                if lip_y < 0:
                    lip_y = 0
                if lip_x < 0:
                    lip_x = 0

                if (lip_y + lip_box) >= frame.shape[0]:
                    lip_y = lip_y - ((lip_y + lip_box) - frame.shape[0])
                    if lip_y < 0:
                        lip_y = 0

                if (lip_x + lip_box) >= frame.shape[1]:
                    lip_x = lip_x - ((lip_x + lip_box) - frame.shape[1])
                    if lip_x < 0:
                        lip_x = 0

                #lip_frame = frame[lip_y:lip_y + lip_box, lip_x:lip_x + lip_box, :]
                """
                # hand segmentation
                hand_x, hand_y, _, _ = hand_point_info
                hand_y = hand_y - 40
                hand_x = hand_x - 40
                if hand_x < 0:
                    hand_x = 0
                if hand_y < 0:
                    hand_y = 0

                if (hand_y + hand_box) > frame.shape[0]:
                    hand_y = hand_y - ((hand_y + hand_box) - frame.shape[0])
                    if hand_y < 0:
                        hand_y = 0
                if (hand_x + hand_box) > frame.shape[1]:
                    hand_x = hand_x - ((hand_x + hand_box) - frame.shape[1])
                    if hand_x < 0:
                        hand_x = 0

                hand_frame = frame[hand_y:hand_y + hand_box, hand_x:hand_x + hand_box, :]

                #lip_frames.append(lip_frame)
                hand_frames.append(hand_frame)
                hand_positions.append([hand_x, hand_y])
                frame_indices.append(frame_idx)
        else:
            videoCapture.release()
            break

    if hand_frames:
        hand_frame_array = np.stack(hand_frames)
    else:
        hand_frame_array = np.empty((0, hand_box, hand_box, 3), dtype=np.uint8)
    return (
        hand_frame_array,
        np.asarray(hand_positions, dtype=np.float32).reshape(-1, 2),
        np.asarray(frame_indices, dtype=np.int64),
    )




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('-p', '--person', type=str, default='M001')
    args = parser.parse_args()
    persons = args.person.split(',')
    segment(args.data_path, persons)


# test(data_path)


# lip = cv2.VideoCapture('/mntnfs/med_data4/cs/test/segmented/LF/lip/0001.mp4')
# frame_counter = int(lip.get(cv2.CAP_PROP_FRAME_COUNT))
# print(frame_counter)


# hand = cv2.VideoCapture('/mntnfs/med_data4/cs/test/segmented/LF/hand/0001.mp4')
# frame_counter = int(hand.get(cv2.CAP_PROP_FRAME_COUNT))
# print(frame_counter)
