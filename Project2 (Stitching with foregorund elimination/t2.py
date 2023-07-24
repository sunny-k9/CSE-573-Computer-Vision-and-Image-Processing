# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades
import json
import cv2
import numpy as np


class DMatcher:
    def __init__(self, idx1, idx2, distance):
        self.idx1 = idx1
        self.idx2 = idx2
        self.distance = distance


def get_features(image):
    sift_obj = cv2.SIFT_create()
    kp, features = sift_obj.detectAndCompute(image, None)
    return kp, features


def match_features(features1, features2):
    dist_matrix = np.zeros((features1.shape[0], features2.shape[0]))
    matches = []
    for i, feature_i in enumerate(features1):
        for j, feature_j in enumerate(features2):
            dist_matrix[i][j] = np.linalg.norm(feature_i - feature_j)

    for i in range(0, dist_matrix.shape[0]):
        min_idx = np.argmin(dist_matrix[i])
        min_idx2 = np.where(dist_matrix[i] == np.partition(dist_matrix[i], 2)[2])[0][0]

        matcher_obj_best = DMatcher(i, min_idx, dist_matrix[i][min_idx])
        matcher_obj_second_best = DMatcher(i, min_idx2, dist_matrix[i][min_idx2])
        matches.append((matcher_obj_best, matcher_obj_second_best))

    return matches


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    kp1, features1 = get_features(img1)
    kp2, features2 = get_features(img2)

    matches = match_features(features1, features2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    matches = good

    ptsimg1 = np.float32([kp1[m.idx1].pt for m in matches])
    ptsimg2 = np.float32([kp2[m.idx2].pt for m in matches])
    if len(ptsimg1) < 4:
        return None, None, None, None
    (H, status) = cv2.findHomography(ptsimg1, ptsimg2, cv2.RANSAC,
                                     4)

    img1_coords = np.array(
        [[0, img1.shape[1], 0, img1.shape[1]], [0, 0, img1.shape[0], img1.shape[0]], [1, 1, 1, 1]])
    warped_pts_img1 = H @ img1_coords

    x_trans = min(warped_pts_img1[0])
    y_trans = min(warped_pts_img1[1])

    if x_trans < 0:
        x_trans = abs(int(x_trans))
    else:
        x_trans = 0
    if y_trans < 0:
        y_trans = abs(int(y_trans))
    else:
        y_trans = 0

    # Combined width height
    x_max = max([img2.shape[1] + x_trans, max(warped_pts_img1[0]) + x_trans])
    y_max = max([img2.shape[0] + y_trans, max(warped_pts_img1[1]) + y_trans])

    width = int(x_max)
    height = int(y_max)

    H[0][2] += x_trans
    H[1][2] += y_trans
    result = cv2.warpPerspective(img1, H, (width, height))
    warped_img1 = result.copy()

    if (img2.shape[0] + y_trans) > height or (img2.shape[1] + x_trans) > width:
        result, warped_img1_mask, result_mask, img2_shifted_mask = None, None, None, None
    else:
        result[y_trans:img2.shape[0] + y_trans, x_trans:img2.shape[1] + x_trans] = img2

        img2_shifted = np.zeros(result.shape, dtype=np.uint8)
        warped_img1_mask = get_mask(warped_img1)
        result_mask = get_mask(result)

        img2_shifted[y_trans:img2.shape[0] + y_trans, x_trans:img2.shape[1] + x_trans] = img2
        img2_shifted_mask = get_mask(img2_shifted)

    return result, warped_img1_mask, result_mask, img2_shifted_mask


def get_mask(img1):
    warped_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    result_bin = cv2.threshold(warped_gray, 1, 255, cv2.THRESH_BINARY)[1]
    return result_bin


def get_overlap_mask(warped_img1_mask, result_mask, img2_shifted_mask):
    overlap_mask = cv2.bitwise_and(warped_img1_mask, img2_shifted_mask)
    return overlap_mask


def stitch(imgmark, N=5,
           savepath=''):  # For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1, N + 1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"

    # Get overlap array
    overlap_arr = np.zeros((N, N), dtype=np.uint8)
    result = None
    for i in range(N - 1):
        height_img1 = imgs[i].shape[0]
        width_img1 = imgs[i].shape[1]

        overlap_arr[i][i] = 1
        for j in range(i + 1, N):
            result, warped_img1_mask, result_mask, img2_shifted_mask = stitch_background(imgs[i], imgs[j])
            if result is not None:
                height_img2 = imgs[j].shape[0]
                width_img2 = imgs[j].shape[1]

                overlap_mask = get_overlap_mask(warped_img1_mask, result_mask, img2_shifted_mask)
                overlap_percent = (np.count_nonzero(overlap_mask) / (
                        (height_img1 * width_img1) + (height_img2 * width_img2))) * 100
                if N == 5:
                    overlap_thresh_val = 20
                else:
                    overlap_thresh_val = 10

                if overlap_percent > overlap_thresh_val:
                    overlap_arr[i][j] = 1
                    overlap_arr[j][i] = 1

    overlap_arr[N - 1][N - 1] = 1

    N, N = overlap_arr.shape

    first_touch = True

    for i in range(N - 1):
        for j in range(i + 1, N):
            if overlap_arr[i][j] == 1:
                if first_touch == True:
                    result, warped_img1_mask, result_mask, img2_shifted_mask = stitch_background(imgs[i], imgs[j])
                    first_touch = False
                result, warped_img1_mask, result_mask, img2_shifted_mask = stitch_background(result, imgs[j])

    cv2.imwrite(savepath, result)

    return overlap_arr


if __name__ == "__main__":
    # task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    # bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
