'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image

import cv2
import numpy as np
import os
import sys
from sklearn.cluster import KMeans

import face_recognition

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path: str) -> dict:
    result_list = []
    '''
    Your implementation.
    '''

    imgs_with_filename = []
    for filename in os.listdir(input_path):
        img = cv2.imread(os.path.join(input_path, filename))
        if img is not None:
            imgs_with_filename.append([img, filename])

    for img, filename in imgs_with_filename:

        model = "opencv_face_detector_uint8.pb"

        config = "opencv_face_detector.pbtxt"

        net = cv2.dnn.readNetFromTensorflow(model, config)
        blob = cv2.dnn.blobFromImage(img, 1.3, (300, 300))

        net.setInput(blob)
        detections = net.forward()

        img_height, img_width, channels = img.shape

        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0][0][i][2]
            if confidence > 0.7:
                x1 = int(detections[0][0][i][3] * img_width)
                y1 = int(detections[0][0][i][4] * img_height)
                x2 = int(detections[0][0][i][5] * img_width)
                y2 = int(detections[0][0][i][6] * img_height)

                bboxes.append([x1, y1, abs(x2 - x1), abs(y2 - y1)])

        for (x1, y1, width, height) in bboxes:
            #cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), (255, 0, 0), 2)
            result_list.append({"iname": filename, "bbox": [x1, y1, width, height]})
        # cv2.imshow("aa",img)
        # cv2.waitKey(0)
    return result_list


'''
K: number of clusters
'''


def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    '''
    Your implementation.
    '''

    detection_result_list = detect_faces(input_path)
    imgs_with_filename = []
    for filename in os.listdir(input_path):
        img = cv2.imread(os.path.join(input_path, filename))
        if img is not None:
            imgs_with_filename.append([img, filename])

    encodings_with_name = []
    for img, filename in imgs_with_filename:
        bboxes_for_single_image = []
        bboxes_for_single_image = [detection["bbox"] for detection in detection_result_list if
                                   detection["iname"] == filename]
        # convert bbox to face_recognition module format
        bboxes_for_single_image_converted = [[bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0]] for bbox in
                                             bboxes_for_single_image]
        encodings = face_recognition.face_encodings(img, bboxes_for_single_image_converted)
        for encoding in encodings:
            encodings_with_name.append((encoding, filename))

    # KMeans
    encodings = list(map(lambda x: x[0], encodings_with_name))
    encodings = np.array(encodings)
    K = int(K)
    kmeans = KMeans(n_clusters=K, algorithm="full").fit(
        encodings)

    cluster_elements = [[] for x in range(K)]
    for idx, label in enumerate(kmeans.labels_):
        cluster_elements[label].append(encodings_with_name[idx][1])

    for cluster_number in range(K):
        result_list.append({"cluster_no": cluster_number, "elements": cluster_elements[cluster_number]})

    # save_cluster_images(imgs_with_filename, result_list)

    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""


def save_cluster_images(imgs_with_filename, result_list):
    for cluster_with_elements in result_list:

        count_h = 0

        comb_image = None
        curr_row = None
        prev_row = None
        pad = np.zeros((100, 100, 3), np.uint8)
        for image_name in cluster_with_elements['elements']:
            img = [img_with_name[0] for img_with_name in imgs_with_filename if
                   img_with_name[1] == image_name][0]
            img = cv2.resize(img, (100, 100))

            if curr_row is None:
                curr_row = img


            else:
                count_h += 1
                curr_row = np.hstack([curr_row, img])
                if count_h == 2:
                    if prev_row is None:
                        prev_row = curr_row

                        curr_row = None
                        count_h = 0
                    else:
                        if comb_image is None:
                            comb_image = prev_row
                        comb_image = np.vstack([comb_image, curr_row])
                        count_h = 0
                        curr_row = None
        pad_flag = False
        if curr_row is not None:
            for i in range(count_h, 2):
                pad_flag = True
                curr_row = np.hstack([curr_row, pad])
            if pad_flag:
                if comb_image is None:
                    comb_image = prev_row
                comb_image = np.vstack([comb_image, curr_row])
        cv2.imwrite("Cluster_" + str(cluster_with_elements["cluster_no"] + 1) + ".jpg", comb_image)
        # cv2.namedWindow("aa", cv2.WINDOW_NORMAL)
        # cv2.imshow("aa", comb_image)
        # cv2.waitKey(0)
