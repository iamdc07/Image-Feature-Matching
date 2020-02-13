import numpy as np
import cv2
import math

h_window = 5
match_threshold = 0.5
h_threshold = 40000000


def load_image():
    # img = cv2.imread("image_sets/graf/img1.ppm", 0)
    # coloured_img = cv2.imread("image_sets/graf/img1.ppm")
    # img2 = cv2.imread("image_sets/graf/img2.ppm", 0)
    # coloured_img2 = cv2.imread("image_sets/graf/img2.ppm")
    # img2 = cv2.imread("image_sets/graf/img4.ppm", 0)
    # coloured_img2 = cv2.imread("image_sets/graf/img4.ppm")
    # img = cv2.imread("image_sets/panorama/pano1_0008.jpg", 0)
    # coloured_img = cv2.imread("image_sets/panorama/pano1_0008.jpg")
    # img2 = cv2.imread("image_sets/panorama/pano1_0009.jpg", 0)
    # coloured_img2 = cv2.imread("image_sets/panorama/pano1_0009.jpg")
    # img = cv2.imread("image_sets/panorama/pano1_0010.jpg", 0)
    # coloured_img = cv2.imread("image_sets/panorama/pano1_0010.jpg")
    # img2 = cv2.imread("image_sets/panorama/pano1_0011.jpg", 0)
    # coloured_img2 = cv2.imread("image_sets/panorama/pano1_0011.jpg")
    # img = cv2.imread("image_sets/box.jpg", 0)
    # coloured_img = cv2.imread("image_sets/box.jpg")
    # img2 = cv2.imread("image_sets/box.jpg", 0)
    # coloured_img2 = cv2.imread("image_sets/box.jpg")
    # img = cv2.imread("image_sets/checks.png", 0)
    # coloured_img = cv2.imread("image_sets/checks.png")
    # img2 = cv2.imread("image_sets/checks.png", 0)
    # coloured_img2 = cv2.imread("image_sets/checks.png")
    # img = cv2.imread("image_sets/x.jpeg", 0)
    # coloured_img = cv2.imread("image_sets/x.jpeg")
    # img2 = cv2.imread("image_sets/x.jpeg", 0)
    # coloured_img2 = cv2.imread("image_sets/x.jpeg")
    img = cv2.imread("image_sets/yosemite/yosemite1.jpg", 0)
    coloured_img = cv2.imread("image_sets/yosemite/yosemite1.jpg")
    img2 = cv2.imread("image_sets/yosemite/yosemite2.jpg", 0)
    coloured_img2 = cv2.imread("image_sets/yosemite/yosemite2.jpg")
    # blurry_merged = np.hstack((graffiti, blur))
    # cv2.imshow('Image', graffiti)
    # cv2.waitKey()

    # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # coloured_img = cv2.rotate(coloured_img, cv2.ROTATE_90_CLOCKWISE)
    # cv2.imshow('', im)
    # cv2.waitKey()
    # exit(0)

    height, width = img.shape

    pad = math.floor(h_window / 2)

    # Image 1
    ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    dst = harris_detector(img, height, width, pad, ix, iy)

    # cv2.imshow('Image1', dst)
    # cv2.waitKey(10000)

    feature_points = local_maximum(dst, height, width)
    print("Local Maximum:", len(feature_points))
    feature_points = adaptive_local_maximum(feature_points)
    print("Adaptive Local Maximum:", len(feature_points))
    # feature_points.sort()

    sift_descriptor1 = sift(img, height, width, feature_points)

    kp1 = []
    # for x, y, d in feature_points:
    #     kp1.append(cv2.KeyPoint(y, x, 1))

    for x, y, d in sift_descriptor1:
        kp1.append(cv2.KeyPoint(y, x, 1))

    cv2.drawKeypoints(coloured_img, keypoints=kp1, outImage=coloured_img, color=(0, 0, 255),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

    cv2.imshow('Image 1', coloured_img)
    cv2.waitKey(10000)

    height, width = img2.shape

    ix = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=5)
    iy = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=5)

    dst = harris_detector(img2, height, width, pad, ix, iy)

    feature_points = local_maximum(dst, height, width)
    print("Local Maximum:", len(feature_points))
    feature_points = adaptive_local_maximum(feature_points)
    print("Adaptive Local Maximum:", len(feature_points))
    # feature_points.sort()

    sift_descriptor2 = sift(img2, height, width, feature_points)

    matches = find_matches(sift_descriptor1, sift_descriptor2)

    kp2 = []
    # for x, y, d in feature_points:
    #     kp2.append(cv2.KeyPoint(y, x, 1))

    for x, y, d in sift_descriptor2:
        kp2.append(cv2.KeyPoint(y, x, 1))

    print("Keypoints1:", len(kp1))
    print("Keypoints2:", len(kp2))
    print("Matches:", len(matches))

    result = cv2.drawMatches(coloured_img, kp1, coloured_img2, kp2, matches, None)

    cv2.drawKeypoints(img2, keypoints=kp2, outImage=coloured_img2, color=(0, 0, 255),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

    cv2.imshow('Image 2', result)
    cv2.waitKey()


def find_matches(descriptor1, descriptor2):
    matches = []
    index1 = 0
    # theindex = 0
    # print("Descriptor 1 ", len(descriptor1))
    # print("Descriptor 2 ", len(descriptor2))

    while index1 < len(descriptor1):
        best_ssd = 10
        s_best_ssd = 10
        y, x, value = descriptor1[index1]

        # print("INDEX1 ", index1)
        index2 = 0
        theindex = -1

        while index2 < len(descriptor2):
            y2, x2, value2 = descriptor2[index2]
            difference = (value - value2)
            difference = difference * difference
            # print("Value1:", value, "Value2:", value2, "Value1 - Value2 ", difference)
            ssd = difference.sum()

            if best_ssd > ssd:
                best_ssd = ssd
                theindex = index2

            index2 += 1

        index2 = 0

        while index2 < len(descriptor2):
            y2, x2, value2 = descriptor2[index2]
            difference = (value - value2) ** 2
            ssd = difference.sum()
            # print("Value1 - Value2 ", ssd)

            if s_best_ssd > ssd:
                if index2 != theindex:
                    s_best_ssd = ssd

            index2 += 1

        # print(best_ssd)
        # exit(0)

        if s_best_ssd != 0:
            ssd_ratio = best_ssd / s_best_ssd
        else:
            index1 += 1
            continue

        # print("Best ssd:", best_ssd)
        # print("2nd best ssd:", s_best_ssd)
        # print("SSD Ratio:", ssd_ratio)

        if ssd_ratio < match_threshold and (0.1 < best_ssd < 0.45):
            print("Best ssd:", best_ssd, " | Second Best ssd ", s_best_ssd, " | SSD ratio ", ssd_ratio)
            each_match = cv2.DMatch(index1, theindex, best_ssd)
            matches.append(each_match)

        index1 += 1

    return matches


def harris_detector(img, height, width, offset, ix, iy):
    count = 0
    dst = np.zeros(img.shape, np.uint)

    ix2 = ix * ix
    ix2_blurred = cv2.GaussianBlur(ix2, (3, 3), 1)

    iy2 = iy * iy
    iy2_blurred = cv2.GaussianBlur(iy2, (3, 3), 1)

    ixiy = ix * iy
    ixiy_blurred = cv2.GaussianBlur(ixiy, (3, 3), 1)

    for y in np.arange(offset, height - offset):
        for x in np.arange(offset, width - offset):
            rx2 = ix2_blurred[y - offset:y + offset + 1, x - offset:x + offset + 1]
            ry2 = iy2_blurred[y - offset:y + offset + 1, x - offset:x + offset + 1]
            rxy = ixiy_blurred[y - offset:y + offset + 1, x - offset:x + offset + 1]

            sum_ix2 = rx2.sum()
            sum_iy2 = ry2.sum()
            sum_ixiy = rxy.sum()

            det = sum_ix2 * sum_iy2 - sum_ixiy * sum_ixiy
            trace = sum_ix2 + sum_iy2

            if trace != 0:
                c = math.floor(det / trace)
                # if c > 70000000:
                if c > h_threshold:
                    dst[y, x] = c
                    count += 1

    print(count, " Feature points detected")

    return dst


def local_maximum(img, height, width):
    feature_points = []

    temp_img = np.pad(img, (1, 1), 'constant', constant_values=(0, 0))

    dst = np.zeros(img.shape, np.uint)

    offset = math.floor(3 / 2)
    # offset = 3

    for y in np.arange(offset, height - offset):
        for x in np.arange(offset, width - offset):
            # print("y ", y, " ", "x ", x, "Value ", temp_img[y, x])

            roi = temp_img[y - offset:y + offset + 1, x - offset:x + offset + 1]

            m = np.amax(roi)
            max_location = np.where(m == roi)

            # print(roi)
            # print(max)
            # print(max_location)

            the_y = max_location[0]
            the_x = max_location[1]

            max_y, max_x = the_y[0], the_x[0]

            if max_y == 1 and max_x == 1:
                dst[y - 1, x - 1] = roi[1, 1]
                # print(dst[y - 1, x - 1], " is largest in window")
                # print(roi[1, 1], " is largest in window")
                # print(max_location[0], " ", max_location[1])
            # else:
            #     dst[y, x] = 0
                # print("Point ", temp_img[y, x])

    # feature_points = [0 if each == 0 else each for each in dst]
    # feature_points = list(set(feature_points))

    for i in np.arange(0, height):
        for j in np.arange(0, width):
            if dst[i, j] != 0:
                feature_points.append((i, j, dst[i, j]))
    # exit(0)
    return feature_points


# def rotation_invariance(orientation_window, magnitude_window):
#     orientation = [0, 0, 0, 0, 0, 0, 0, 0]
#     orientation_dict = {0: 0, 1: 45, 2: 90, 3: 135, 4: 180, 5: 225, 6: 270, 7: 315}
#     # orientation_window = np.degrees(orientation_window)
#
#     for x in range(0, 16):
#         for y in range(0, 16):
#             orientation_window[x, y] = math.degrees(orientation_window[x, y])
#             if orientation_window[x, y] < 0:
#                 orientation_window[x, y] += 360
#             elif orientation_window[x, y] > 360:
#                 orientation_window[x, y] = orientation_window[x, y] % 360
#
#             index = math.floor(orientation_window[x, y] / 45)
#             # print(orientation_window[x, y], index)
#
#             orientation[index] += 1
#
#     max_value = max(orientation)
#     max_index = orientation.index(max_value)
#
#     for x in range(0, 16):
#         for y in range(0, 16):
#             orientation_window[x, y] = orientation_window[x, y] - orientation_dict.get(max_index)
#             if orientation_window[x, y] < 0:
#                 orientation_window[x, y] += 360
#
#     # dominant_orientation = orientation_dict.get(max_index)
#
#     orientation_window = np.radians(orientation_window)
#
#     return orientation_window


def rotation_invariance(orientation_window, magnitude_window):
    orientation_window = np.array(orientation_window)
    feature_angle = orientation_window[8, 8]
    for y in np.arange(0, 16):
        for x in np.arange(0, 16):
            if x == 8 and y == 8:
                continue
            angle = orientation_window[y, x]
            orientation_window[y, x] = angle - feature_angle

    return orientation_window


def adaptive_local_maximum(feature_points):
    final = []
    i = 0

    for x, y, c in feature_points:
        i += 1
        min_distance = 999999999

        for x_r, y_r, c_r in feature_points:
            if x == x_r and y == y_r:
                continue
            if c < (0.9 * c_r):
                d = math.sqrt(((y - x) ** 2) + ((y_r - x_r) ** 2))

                if d < min_distance:
                    min_distance = d

        final.append((x, y, min_distance))

    # print(" KeyPoints detected!")

    final.sort(key=lambda p: p[2])

    return final[:270]

    # temp_img = np.pad(img, (1, 1), 'constant', constant_values=(0, 0))
    #
    # dst = np.zeros(img.shape, np.uint)
    #
    # for y, x, value in feature_points:
    #     if (x - 8 < 0 or y - 8 < 0) or (x + 8 > width or y + 8 > height):
    #         continue
    #     else:
    #         window =


def sift(img, height, width, feature_points):
    # Consider rotation invariance in this function
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    magnitude = np.zeros((height, width), np.float)
    orientation = np.zeros((height, width), np.float)
    sift_descriptor = []

    # Pad the image with 1 px border on both axis
    padding = np.pad(img, (1, 1), 'constant', constant_values=(0, 0))
    new_height, new_width = padding.shape
    # print(img.shape)
    # print(new_height)
    # print(new_width)
    # print(padding)
    # exit(0)

    # Finding magnitude and orientation of each pixel
    for y in np.arange(1, new_height - 1):
        for x in np.arange(1, new_width - 1):
            # print("y ", y, " ", "x ", x, "Value ", padding[y, x])
            l2 = int(padding[y + 1, x]) - int(padding[y - 1, x])
            l1 = int(padding[y, x + 1]) - int(padding[y, x - 1])
            # print("L1 ", l2, " ", "L2 ", l2)

            magnitude[y - 1, x - 1] = math.sqrt((l2 * l2) + (l1 * l1))
            orientation[y - 1, x - 1] = math.atan2(l2, l1)

    for y, x, value in feature_points:
        # print("y ", y, " ", "x ", x)
        if (x - 8 < 0 or y - 8 < 0) or (x + 8 > width or y + 8 > height):
            continue
        else:
            magnitude_window = magnitude[y - 8:y + 8, x - 8:x + 8]
            orientation_window = orientation[y - 8:y + 8, x - 8:x + 8]
            if y == 116 and x == 675:
                print()

            orientation_window = rotation_invariance(orientation_window, magnitude_window)

            # print("Reg:", magnitude_window)
            magnitude_window = cv2.normalize(magnitude_window, None, norm_type=cv2.NORM_L2)
            # print("Window ", y, "W:", x, "Val:", magnitude[y, x])
            # print("NORM:", magnitude_window)
            each_descriptor = []
            for i in range(0, 16, 4):
                for j in range(0, 16, 4):
                    orientation_hist = calculate_grid_histogram(magnitude_window[i:i + 4, j:j + 4],
                                                                orientation_window[i:i + 4, j:j + 4])
                    # THRESHOLD & THEN NORMALIZE
                    # orientation_hist = np.array(orientation_hist).reshape(-1)

                    orientation_hist = [0.2 if each > 0.2 else each for each in orientation_hist]
                    # print("NORMALIZED:", orientation_hist[0:10])
                    each_descriptor.append(orientation_hist)

            normalized = cv2.normalize(np.array(each_descriptor).reshape(-1), None, norm_type=cv2.NORM_L2)

            sift_descriptor.append((y, x, normalized))
            # print("DESCRIPTOR ", sift_descriptor)

    # exit(0)

    return sift_descriptor


def calculate_grid_histogram(magnitude_grid, orientation_grid):
    orientation_hist = [0, 0, 0, 0, 0, 0, 0, 0]
    height, width = magnitude_grid.shape

    # print(magnitude_grid)

    for y in np.arange(height):
        for x in np.arange(width):
            degrees = math.degrees(orientation_grid[y, x])
            # print("D:", degrees)
            if degrees < 0:
                degrees += 360
            elif degrees > 360:
                degrees %= 360

            index = math.floor(degrees / 45)

            # print("INDEX:", index, "D:", degrees)
            orientation_hist[index] += magnitude_grid[y, x]

    # print("Before ", orientation_hist)

    return orientation_hist


if __name__ == '__main__':
    load_image()
