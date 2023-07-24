"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    characters_zoning_count = enrollment(characters)

    label_bboxes = detection(test_img)

    return recognition(label_bboxes, characters_zoning_count, test_img)


def crop_character(curr_character):
    height, width = curr_character.shape
    x1, y1, x2, y2 = 9999, 9999, 0, 0
    for i in range(height):
        for j in range(width):
            if curr_character[i][j] == 255:
                x1 = min(x1, j)
                y1 = min(y1, i)
                x2 = max(x2, j)
                y2 = max(y2, i)
    return curr_character[y1:(y2 + 1), x1:(x2 + 1)]


def get_zoning_matrix(curr_character, zone_size, char_size):
    height, width = curr_character.shape

    # if height >= width:
    #     ratio = width/height
    #     dim = (int(ratio*40),40)
    # else:
    #     ratio = height/width
    #     dim = (40,int(ratio*40))
    # resized_img = cv2.resize(np.uint8(255*curr_character), dim, interpolation = cv2.INTER_AREA)
    # height , width = np.shape(resized_img)
    # padded_img = np.zeros((40,40))
    # u = int((40 - height)/2)
    # v = int((40 - width)/2)
    # padded_img[u:u+height,v:v+width] = resized_img
    curr_character = cv2.resize(curr_character, char_size)

    zone_pixel_length_in_image = int(char_size[0] / zone_size[0])

    zone_count_values = np.zeros((zone_size[0], zone_size[0]))
    for i in range(zone_size[0]):
        for j in range(zone_size[1]):
            zone_img = curr_character[(zone_pixel_length_in_image * i):(zone_pixel_length_in_image * (i + 1)),
                       (zone_pixel_length_in_image * j):(zone_pixel_length_in_image * (j + 1))]
            white_count = np.count_nonzero(zone_img)

            zone_count_values[i][j] = white_count

    return zone_count_values


def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    characters_zoning_counts = []
    for idx, character in enumerate(characters):
        character_name = character[0]
        curr_character = 255 - character[1]

        char_size = (20, 20)
        zone_size = (5, 5)
        curr_character = crop_character(curr_character)

        zone_count_values = get_zoning_matrix(curr_character, zone_size, char_size)

        characters_zoning_counts.append([character_name, zone_count_values])
        # cv2.imshow("a", curr_character)
        # cv2.waitKey(0)
    # for character_zoning_ratio in characters_zoning_counts:
    #     print(character_zoning_ratio[0])
    #     print(character_zoning_ratio[1])

    return characters_zoning_counts


def detection(test_img):
    """
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    height, width = test_img.shape
    thresholded_img = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            if test_img[i][j] < 200:
                thresholded_img[i][j] = 1
            else:
                thresholded_img[i][j] = 0

    label_img = thresholded_img.copy()
    label_count = 0

    conflict_loc = {}

    # First Pass
    for i in range(1, height):
        for j in range(1, width):
            if thresholded_img[i][j] == 1:
                # in_flag =True
                n_neighbour = label_img[i - 1][j]
                w_neighbour = label_img[i][j - 1]

                if n_neighbour > 0:
                    if w_neighbour > 0:
                        label_img[i][j] = min(n_neighbour, w_neighbour)
                        if n_neighbour > label_img[i][j]:

                            if n_neighbour in conflict_loc:
                                link_flag = False
                                backtrack_idx = n_neighbour
                                while True:
                                    temp_backtrack_idx = conflict_loc[backtrack_idx]

                                    if temp_backtrack_idx in conflict_loc:
                                        backtrack_idx = conflict_loc[backtrack_idx]

                                        link_flag = True
                                    else:
                                        break
                                if link_flag:
                                    conflict_loc[n_neighbour] = min(conflict_loc[backtrack_idx], label_img[i][j])
                                else:
                                    conflict_loc[n_neighbour] = min(conflict_loc[n_neighbour], label_img[i][j])
                            else:
                                conflict_loc[n_neighbour] = label_img[i][j]
                        if w_neighbour > label_img[i][j]:

                            if w_neighbour in conflict_loc:
                                link_flag = False
                                backtrack_idx = w_neighbour
                                while True:
                                    temp_backtrack_idx = conflict_loc[backtrack_idx]

                                    if temp_backtrack_idx in conflict_loc:
                                        backtrack_idx = conflict_loc[backtrack_idx]

                                        link_flag = True
                                    else:
                                        break
                                if link_flag:
                                    conflict_loc[w_neighbour] = min(conflict_loc[backtrack_idx], label_img[i][j])
                                else:
                                    conflict_loc[w_neighbour] = min(conflict_loc[w_neighbour], label_img[i][j])

                            else:
                                conflict_loc[w_neighbour] = label_img[i][j]




                    else:
                        label_img[i][j] = n_neighbour
                elif w_neighbour > 0:
                    label_img[i][j] = w_neighbour
                else:
                    # if in_flag == True:
                    label_count += 1
                    label_img[i][j] = label_count
                    # in_flag = False
    # Second pass
    label_bboxes = {}

    for i in range(1, height):
        for j in range(1, width):
            curr_label = label_img[i][j]
            if curr_label > 0:
                if curr_label in conflict_loc:
                    curr_label = conflict_loc[curr_label]
                    label_img[i][j] = curr_label

                if curr_label in label_bboxes:
                    label_bboxes[curr_label] = (
                        min(label_bboxes[curr_label][0], j), min(label_bboxes[curr_label][1], i),
                        max(label_bboxes[curr_label][2], j), max(label_bboxes[curr_label][3], i))
                else:
                    label_bboxes[curr_label] = (j, i, j, i)

    for key in list(label_bboxes):
        area = np.abs((label_bboxes[key][2] - label_bboxes[key][0]) * (label_bboxes[key][3] - label_bboxes[key][1]))
        if area < 30:
            del (label_bboxes[key])
        # print(str(key) + ": " + str(area))

    colored_img = getColoredImage(label_img)
    bbox_image = colored_img.copy()

    for key in label_bboxes:
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 1
        bbox_image = cv2.rectangle(bbox_image, (label_bboxes[key][0], label_bboxes[key][1]),
                                   (label_bboxes[key][2], label_bboxes[key][3]), color, thickness)


    return label_bboxes


def getColoredImage(label_img):
    height, width = label_img.shape
    color_map = {}
    colored_img = np.zeros((height, width, 3), np.uint8)

    for i in range(1, height):
        for j in range(1, width):
            if label_img[i][j] > 0:
                if label_img[i][j] in color_map:
                    colored_img[i][j] = color_map[label_img[i][j]]
                else:
                    color_map[label_img[i][j]] = (np.random.randint(0, 255), np.random.randint(0, 255),
                                                  np.random.randint(0, 255))
                    colored_img[i][j] = color_map[label_img[i][j]]
    return colored_img


def recognition(label_bboxes, characters_zoning_count, test_img):
    """
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    distance_scores = []
    char_size = (20, 20)
    zone_size = (5, 5)
    det_count = 0
    result = []
    thresh = 300
    for key in label_bboxes:
        detected_char = test_img[label_bboxes[key][1]:(label_bboxes[key][3] + 1),
                        label_bboxes[key][0]:(label_bboxes[key][2] + 1)].copy()
        detected_char = cv2.resize(detected_char, (char_size[0], char_size[0]))
        min_SSD = 999999
        min_char = None
        detected_char = 255 - detected_char
        zone_count_values = get_zoning_matrix(detected_char, zone_size, char_size)

        for character_with_zoning_count in characters_zoning_count:
            SSD = np.power((zone_count_values - character_with_zoning_count[1]), 2)
            SSD = np.sum(SSD)
            if SSD < min_SSD:
                min_SSD = SSD
                min_char = character_with_zoning_count[0]

        if min_SSD < thresh:
            result.append({"bbox": list(label_bboxes[key]), "name": min_char})
        else:
            result.append({"bbox": list(label_bboxes[key]), "name": "UNKOWN"})
        det_count += 1

    print(result)
    return result


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    characters = []

    all_character_imgs = glob.glob(args.character_folder_path + "/*")

    for each_character in all_character_imgs:
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
