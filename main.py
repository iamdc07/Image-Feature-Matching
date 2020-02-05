import numpy as np
import cv2
import math

h_window = 5

def load_image():
    img = cv2.imread("image_sets/graf/img2.ppm", 0)
    coloured_img = cv2.imread("image_sets/graf/img2.ppm")
    # img = cv2.imread("image_sets/yosemite/yosemite1.jpg", 0)
    # coloured_img = cv2.imread("image_sets/yosemite/yosemite1.jpg")
    # blurry_merged = np.hstack((graffiti, blur))
    # cv2.imshow('Image', graffiti)
    # cv2.waitKey()

    height, width = img.shape

    pad = math.floor(h_window/2)

    ix, iy = calculate_gradient(img)

    dst = harris_detector(img, coloured_img, height, width, pad, ix, iy)

    feature_points = local_maximum(dst, height, width)

    kp1 = []
    for x, y, d in feature_points:
        kp1.append(cv2.KeyPoint(x, y, 1))

    cv2.drawKeypoints(coloured_img, keypoints=kp1, outImage=coloured_img,  color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

    cv2.imshow('Keypoints ', coloured_img)
    cv2.waitKey(0)


def calculate_gradient(img):
    # sobel_x = np.array([[1, 0, -1],
    #                     [2, 0, -2],
    #                     [1, 0, -1]])
    #
    # sobel_y = np.array([[1, 2, 1],
    #                     [0, 0, 0],
    #                     [-1, -2, -1]])
    #
    # ix = cv2.filter2D(img, -1, sobel_x)
    # iy = cv2.filter2D(img, -1, sobel_y)

    ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    return ix, iy


def harris_detector(img, coloured_img, height, width, pad, ix, iy):
    dst = np.zeros(img.shape, np.uint)

    ix2 = ix * ix
    ix2_blurred = cv2.GaussianBlur(ix2, (3, 3), 1)

    iy2 = iy * iy
    iy2_blurred = cv2.GaussianBlur(iy2, (3, 3), 1)

    ixiy = ix * iy
    ixiy_blurred = cv2.GaussianBlur(ixiy, (3, 3), 1)

    for y in np.arange(pad, height - pad):
        for x in np.arange(pad, width - pad):
            rx2 = ix2_blurred[y - pad:y + pad + 1, x - pad:x + pad + 1]
            ry2 = iy2_blurred[y - pad:y + pad + 1, x - pad:x + pad + 1]
            rxy = ixiy_blurred[y - pad:y + pad + 1, x - pad:x + pad + 1]

            sum_ix2 = rx2.sum()
            sum_iy2 = ry2.sum()
            sum_ixiy = rxy.sum()

            det = sum_ix2*sum_iy2 - sum_ixiy*sum_ixiy
            trace = sum_ix2 + sum_iy2

            if trace != 0:
                c = math.floor(det / trace)
                if c > 50000000:
                    dst[y, x] = c


    # print(dst)

    # product_1 = np.multiply(ix2_blurred, iy2_blurred).astype(np.uint8)
    # product_2 = np.multiply(ixiy_blurred, ixiy_blurred).astype(np.uint8)
    #
    # det = np.subtract(product_1, product_2)
    #
    # trace = np.add(ix2_blurred, iy2_blurred).astype(np.uint8)
    #
    # c = np.divide(det, trace).astype(np.uint8)
    #
    # rows, columns = img.shape
    #
    # dst1 = np.zeros(c.shape, np.uint8)
    #
    # dst = cv2.filter2D(c, ddepth=-1, kernel=kernel)
    #
    # print(img.shape)
    #
    # i = 0
    #
    # while i < rows:
    #     j = 0
    #     while j < columns:
    #         if dst[i, j] > 100:
    #             dst1[i, j] = dst[i, j]
    #         else:
    #             dst1[i, j] = 0
    #
    #         j += 1
    #     i += 1
    #
    # i = 0
    #
    # while i < rows:
    #     j = 0
    #     while j < columns:
    #         if dst1[i, j] > 0:
    #             coloured_img.itemset((i, j, 1), 255)
    #             coloured_img.itemset((i, j, 2), 0)
    #             coloured_img.itemset((i, j, 0), 0)
    #
    #         j += 1
    #     i += 1

    # c2 = np.divide(det, trace).astype(np.uint8)

    # merged = np.hstack((dst, coloured_img))

    # cv2.imshow('', dst)
    # cv2.waitKey()
    return dst


def local_maximum(img, height, width):
    feature_points = []

    temp_img = np.pad(img, (1, 1), 'constant', constant_values=(0, 0))

    pad = math.floor(3 / 2)

    for y in np.arange(pad, height - pad):
        for x in np.arange(pad, width - pad):
            roi = temp_img[y - pad:y + pad + 1, x - pad:x + pad + 1]

            max = np.amax(roi)
            max_location = np.where(max == roi)

            # print(roi)
            # print(max)
            # print(max_location)

            max_y = max_location[0]
            max_x = max_location[1]

            py, px = max_y[0], max_x[0]

            if py == 1 and px == 1:
                feature_points.append((x - 1, y - 1, temp_img[1, 1]))
                # print(temp_img[y, x], " is largest in window")
                # print(max_location[0], " ", max_location[1])
            else:
                temp_img[y, x] = 0
                # print("Point ", temp_img[y, x])


    # print(feature_points)
    feature_points = list(set(feature_points))

    return feature_points


if __name__ == '__main__':
    load_image()
