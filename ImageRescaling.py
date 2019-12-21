import inspect
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def myHistEqualization(Img):
    HistL = np.zeros(256)
    for i in range(256):
        HistL[i] = len(Img[Img == np.float64(i)])

    PDFfun = HistL / np.sum(HistL)

    CDFfun = np.zeros(PDFfun.shape)
    CDFfun[0] = PDFfun[0]
    for k in range(1, len(CDFfun)):
        CDFfun[k] = CDFfun[k - 1] + PDFfun[k]

    RepImg = np.float64(Img)
    for j in range(256):
        RepImg[RepImg == j] = CDFfun[j] * 255

    return np.uint8(np.round(RepImg))


# Arrange the pixels in the zoom-in image
def scatter(crop_img, ratio):
    crop_img = np.asarray(crop_img, np.uint8)

    # Create a new rows and cols according to the crop image and the zoom-in ratio:
    rowsSize = int(crop_img.shape[0] * ratio)
    colsSize = int(crop_img.shape[1] * ratio)
    new_img = np.zeros((rowsSize, colsSize), np.uint8)

    # Scanning the rows and columns and fill the matrix according to the this calculation:
    dim_crop_img = crop_img.shape
    for i in range(0, dim_crop_img[0]):
        for j in range(0, dim_crop_img[1]):
            new_img[int(i * ratio) - 1, int(j * ratio) - 1] = crop_img[i, j]  # Arranges the pixels according to the

    return new_img


# The function get as input: crop image, crop image is scattered on a large matrix according to the ratio and order
# of interpolation, And calculates the interpolation in nonlinear calculation by cubic interpolation:
def inter(imgcrop, zoomin, ratio):
    imgx = np.asarray(zoomin, np.float)
    imgy = np.asarray(zoomin, np.float)

    x = np.linspace(0, imgcrop.shape[1], imgcrop.shape[1], True)
    xnew = np.linspace(0, imgcrop.shape[1], imgx.shape[1], True)

    j = 0

    # loop that calculates interpolation by y:
    for i in range(imgcrop.shape[0]):
        y = imgcrop[i, :]
        f2 = interp1d(x, y, 'cubic')
        d = np.asarray(f2(xnew), np.float)
        imgx[int(j * ratio) - 1, :] = d
        j = j + 1

    y = np.linspace(0, imgcrop.shape[0], imgcrop.shape[0], True)
    ynew = np.linspace(0, imgcrop.shape[0], imgx.shape[0], True)

    # loop that calculates interpolation by x:
    j = 0
    for i in range(imgcrop.shape[1]):
        x = imgcrop[:, i]

        f2 = interp1d(y, x, 'cubic')
        d = np.asarray(f2(ynew), np.float)
        imgy[:, int(j * ratio) - 1] = d
        j = j + 1
    # Merge the interpolation of x and y:
    totalImage = (imgy + imgx) / 0.8
    totalImage[totalImage > 255] = 255

    # Takes the original values of the matrix before the interpolation
    for i in range(imgcrop.shape[0]):
        for j in range(imgcrop.shape[1]):
            totalImage[int(i * ratio) - 1, int(j * ratio) - 1] = zoomin[int(i * ratio) - 1, int(j * ratio) - 1]

    totalImage = np.asarray(totalImage, np.uint8)
    return totalImage


# The function selects the part we want to zoom-in, scatter the pixels in the new large matrix and interpolates:
def resize(img, ratio):
    # Checks if the input is correct:
    if img is None:
        print ("the image  is empty")
        return None
    if ratio < 1:
        ratio = 1

    # Choosing the part you want to zoom-in:
    r = cv2.selectROI(img)
    crop_img = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

    if len(crop_img.shape) == 3:
        # if the image is colored, split the matrix to: b,g,r
        b = crop_img[:, :, 0]
        g = crop_img[:, :, 1]
        r = crop_img[:, :, 2]
        # Scatter the split image With the scatter function:
        newb = scatter(b, ratio)
        newg = scatter(g, ratio)
        newr = scatter(r, ratio)
        # Interpolate the split image:
        b = inter(b, newb, ratio)
        g = inter(g, newg, ratio)
        r = inter(r, newr, ratio)

        # Merge the three matrix into one image:
        fin = cv2.merge((b, g, r))

    else:  # If the image is grayscale:
        # Scatter the image With the scatter function and interpolate the image
        newgray = scatter(crop_img, ratio)
        fin = inter(crop_img, newgray, ratio)

    fin = cv2.GaussianBlur(fin, (5, 5), 0.4)
    fin = myHistEqualization(fin)
    plt.figure()
    plt.subplot(121)
    plt.title('src image')
    plt.imshow(img[:, :, ::-1])

    plt.subplot(122)

    plt.imshow(fin[:, :, ::-1])
    plt.title('zoom with interpolation')
    plt.show()


# The function returns a 2x smaller image
def Gaus(img):
    # Checks if the input is correct:
    if img is None:
        print('Error opening image!')
        return -1

    if len(img.shape) == 3:
        # if the image is colored, split the matrix and call each split matrix again
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]

        b = Gaus(b)
        g = Gaus(g)
        r = Gaus(r)

        # Merge the three matrix into one image:
        newimg = cv2.merge((b, g, r))
        newimg = np.asarray(newimg, np.uint8)
        return newimg

    else:
        # If the image is grayscale:
        im = img.copy()

        # Reduce image size by 2
        w = int(im.shape[0] / 2)
        h = int(im.shape[1] / 2)
        n = 0
        m = 0
        new_im = np.zeros((w, h), np.uint8)

        # Loop run on the rows and columns of the new image and fill it with the following calculation:
        for i in range(new_im.shape[0]):
            n = i * 2
            for j in range(new_im.shape[1]):
                m = j * 2
                new_im[i, j] = int((np.sum(im[n:n + 2, m:m + 2])) / 4)

        im = cv2.GaussianBlur(new_im, (5, 5), 0.4)
        return im


# The function creates a pyramid of images that are 2 times smaller in each level:
def gaussianPyramid(img, level):
    # Checks if the input is correct:
    if img is None:
        print ('Error opening image!')
        return -1
    if level > 9:
        print ('Error, Maximum size of level pyramid is 9!')
        return -1
    # Loop by level and show the next level image in the pyramid:
    im = img.copy()
    for k in range(level):
        im = cv2.GaussianBlur(im, (5, 5), 0.4)
        cv2.imshow("pyramid", im)
        new_im = Gaus(im)
        im = new_im
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    img = cv2.imread("Cat.jpg")

    # gaussianPyramid(img, 4)

    resize(img, 2)


main()
