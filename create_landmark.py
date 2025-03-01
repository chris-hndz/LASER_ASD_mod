import glob
import os
import tqdm
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import json

import warnings
warnings.filterwarnings('ignore')

os.environ["GRPC_VERBOSITY"] = "NONE"

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
LIGHT_GREEN_COLOR = (144, 238, 144)
WHITE_YELLOW_COLOR = (255, 255, 224)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--l', type = int, required=True)
parser.add_argument('--r', type = int, required=True)
parser.add_argument('--type', type = str, required=True)
args = parser.parse_args()

path_to_frame = f'ASW_Dataset/clips_videos/{args.type}'
out_dir = f'ASW_Dataset/clip_videos_landmark/{args.type}'
folders = os.listdir(path_to_frame)
folders = sorted(folders)

# lips = [
#     61, 291,

#     405, 17, 181,

#     80, 13, 310
# ]

lips = [
        # lipsUpperOuter
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        # lipsLowerOuter in reverse order
        375, 321, 405, 314, 17, 84, 181, 91, 146, 61,

        76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 306,

        307, 320, 404, 315, 16, 85, 180, 90, 77,

        62, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292,

        325, 319, 403, 316, 15, 86, 179, 89, 96,

        #  lipsUpperInner
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
        # lipsLowerOuter in reverse order
            324, 318, 402, 317, 14, 87, 178, 88, 95, 78
]


presence_threshold = solutions.drawing_utils._PRESENCE_THRESHOLD
visibility_threshold = solutions.drawing_utils._VISIBILITY_THRESHOLD

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

json_path = f'ASW_Dataset/csv/landmark/{args.type}'


cnt = 0
total = 0

base_options = python.BaseOptions(model_asset_path='weights/face_landmarker_v2_with_blendshapes.task', delegate = python.BaseOptions.Delegate.GPU)
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                output_face_blendshapes=True,
                                                output_facial_transformation_matrixes=True,
                                                num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

for folder in tqdm.tqdm(folders[args.l:min((args.r+1), len(folders)):1]):
    videoName = folder
    if os.path.exists(os.path.join(out_dir, folder)) == False:
        os.makedirs(os.path.join(out_dir, folder))
    os.makedirs(os.path.join(json_path, folder), exist_ok=True)
    subfolders = glob.glob(os.path.join(path_to_frame, folder + '/*'))
    for subfolder in subfolders:
        if os.path.exists(os.path.join(out_dir, folder, subfolder.split('/')[-1])) == False:
            os.makedirs(os.path.join(out_dir, folder, subfolder.split('/')[-1]))
        files = glob.glob(subfolder + '/*')

        landmark_json = {}
        files = sorted(files)

        for id, file in enumerate(files):

            total += 1
            writeDir = os.path.join(out_dir, folder, subfolder.split('/')[-1], file.split('/')[-1])
            time = file.split('/')[-1][:-4]
            landmark_json[time] = []

            cv_mat = cv2.imread(file)
            cv_mat = cv2.resize(cv_mat, (128, 128))
            

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)

            detection_result = detector.detect(image)
            
            image = image.numpy_view()
            black_image = np.zeros((128, 128, 3), dtype=np.uint8)
            H, W, _ = image.shape

            landmark_lists = []

            for i in range(len(detection_result.face_landmarks)):
                landmark_lists.extend(detection_result.face_landmarks[i])
            ok = 0
            for id in lips:
                if id >= len(landmark_lists):
                    landmark_json[time].append((-1, -1))
                    continue
                landmark = landmark_lists[id]
                # print(landmark)
                landmark_px = solutions.drawing_utils._normalized_to_pixel_coordinates(landmark.x, landmark.y, W, H)
                # print(landmark_px)

                if landmark_px:
                    ok = 1
                    image = cv2.circle(np.float32(image), landmark_px, radius = 1, color = LIGHT_GREEN_COLOR, thickness=2)
                    black_image = cv2.circle(np.float32(black_image), landmark_px, radius = 1, color = LIGHT_GREEN_COLOR, thickness=2)
                    landmark_json[time].append((landmark.x, landmark.y))
                else:
                    landmark_json[time].append((-1, -1))
        
            assert(len(landmark_json[time]) == 82)
            if ok:
                cnt += 1
        
            cv2.imwrite(writeDir, image)

            writeDir = os.path.join(out_dir, folder, subfolder.split('/')[-1], file.split('/')[-1][:4] + "_black.jpg")
            cv2.imwrite(writeDir, black_image)

        ts_f = os.path.join(json_path, folder, subfolder.split('/')[-1] + '.json')
        
        with open(ts_f, 'w') as f:
            json.dump(landmark_json, f)
            # break
        # break
    # break

print("total: ", total)
print("marked faces: ", cnt)
print("unmarked faces: ", total - cnt)