import numpy as np
import cv2


def load_image():
    graffiti = cv2.imread("image_sets/graf/img2.ppm", 0)
    # blurry_merged = np.hstack((graffiti, blur))
    # cv2.imshow('Image', graffiti)
    # cv2.waitKey()
    harris_detector(graffiti)


def harris_detector(img):
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])

    # kernel2 = np.array([[0, 0, 0],
    #                     [0, 1, 0],
    #                     [0, 0, 0]])

    Ix = cv2.filter2D(img, -1, sobel_x)
    Iy = cv2.filter2D(img, -1, sobel_y)

    Ix2 = Ix * Ix
    Ix2_blurred = cv2.GaussianBlur(Ix2, (3, 3), 0)

    Iy2 = Iy * Iy
    Iy2_blurred = cv2.GaussianBlur(Iy2, (3, 3), 0)

    IxIy = Ix * Iy
    IxIy_blurred = cv2.GaussianBlur(IxIy, (3, 3), 0)

    print(img.dtype)

    product_1 = np.multiply(Ix2_blurred, Iy2_blurred).astype(np.uint8)
    product_2 = np.multiply(IxIy_blurred, IxIy_blurred).astype(np.uint8)

    det = np.subtract(product_1, product_2)

    trace = np.add(Ix2_blurred, Iy2_blurred).astype(np.uint8)

    c = np.divide(det, trace, where=trace != 0).astype(np.uint8)

    rows, columns = img.shape

    dst = np.zeros(c.shape, np.uint8)

    i = 0

    while i < rows:
        j = 0
        while j < columns:
            if c[i, j] > 128:
                dst[i, j] = c[i, j]
            else:
                dst[i, j] = 0

            j += 1
        i += 1

    # c2 = np.divide(det, trace).astype(np.uint8)

    # blurry_merged = np.hstack((c, c2))

    cv2.filter2D(dst, ddepth=-1, kernel=kernel)

    cv2.imshow('', dst)
    cv2.waitKey()

    print(c.dtype)


if __name__ == '__main__':
    load_image()
