import numpy as np
import cv2
import math

h_window = 5

def load_image():
    img = cv2.imread("image_sets/graf/img1.ppm", 0)
    coloured_img = cv2.imread("image_sets/graf/img1.ppm")
    # img = cv2.imread("image_sets/yosemite/yosemite1.jpg", 0)
    # coloured_img = cv2.imread("image_sets/yosemite/yosemite1.jpg")
    # blurry_merged = np.hstack((graffiti, blur))
    # cv2.imshow('Image', graffiti)
    # cv2.waitKey()

    height, width = img.shape

    pad = math.floor(h_window/2)

    ix, iy = calculate_gradient(img)

    dst = harris_detector(img, coloured_img, height, width, pad, ix, iy)
    


def calculate_gradient(img):
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    ix = cv2.filter2D(img, -1, sobel_x)
    iy = cv2.filter2D(img, -1, sobel_y)

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
                if c > 10000000:
                    dst[y, x] = c


    print(img.dtype)

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


if __name__ == '__main__':
    load_image()
