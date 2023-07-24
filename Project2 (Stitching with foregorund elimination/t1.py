# Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

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


def get_mask(img1):
    warped_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    result_bin = cv2.threshold(warped_gray, 1, 255, cv2.THRESH_BINARY)[1]
    return result_bin


def get_overlap_mask(warped_img1_mask, img2_shifted_mask):
    overlap_mask = cv2.bitwise_and(warped_img1_mask, img2_shifted_mask)

    return overlap_mask


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


def remove_foregorund(result, result_img1_on_top):
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    result_img1_on_top_gray = cv2.cvtColor(result_img1_on_top, cv2.COLOR_BGR2GRAY)

    (T, threshInv_result) = cv2.threshold(result_gray, 50, 255,
                                          cv2.THRESH_BINARY)
    (T, threshInv_result_img1_on_top) = cv2.threshold(result_img1_on_top_gray, 80, 255,
                                                      cv2.THRESH_BINARY)

    mask_foreground = threshInv_result_img1_on_top - threshInv_result
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

    mask_foreground_eroded = cv2.erode(mask_foreground, kernel, iterations=2)
    mask_foreground_dilated = cv2.dilate(mask_foreground_eroded, kernel, iterations=60)

    foreground_replaced_img = np.zeros((result.shape[0], result.shape[1], result.shape[2]), dtype=np.uint8)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if mask_foreground_dilated[i][j] == 255:
                foreground_replaced_img[i][j] = result_img1_on_top[i][j]
            else:
                foreground_replaced_img[i][j] = result[i][j]
    return foreground_replaced_img


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
    (H, status) = cv2.findHomography(ptsimg1, ptsimg2, cv2.RANSAC,
                                     4)

    img1_coords = np.array(
        [[0, img1.shape[1], 0, img1.shape[1]], [0, 0, img1.shape[0], img1.shape[0]], [1, 1, 1, 1]])
    warped_pts_img1 = H @ img1_coords

    #Calculate x and y translation in the warped image to fit fully in image
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

        img2_shifted[y_trans:img2.shape[0] + y_trans, x_trans:img2.shape[1] + x_trans] = img2

        result_img1_on_top = np.zeros(result.shape, dtype=np.uint8)
        # Overlap 1 on top
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if np.mean(warped_img1[i][j]) == 0:
                    result_img1_on_top[i][j] = img2_shifted[i][j]
                else:
                    result_img1_on_top[i][j] = warped_img1[i][j]

        # remove foreground
        foreground_replaced_img = remove_foregorund(result, result_img1_on_top)

        cv2.imwrite(savepath, foreground_replaced_img)
    return


if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)
