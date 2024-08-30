import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # to read mask

from scipy.ndimage import gaussian_filter


# IMAGE CALCULATIONS

# Select an image with ROI from given image
def select_image_from(source_image):
    # selecting ROI
    roi = cv2.selectROI('Select Image', source_image, fromCenter=False, showCrosshair=True)
    # crops the selected ROI from the image
    selected_image = source_image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    return selected_image


# Finding the last depth gaussian can achieve
def find_pyramid_depth(image):
    depth = 0
    y = image.shape[0]
    x = image.shape[1]
    # loops until one axis is smaller than 2
    while x >= 2 and y >= 2:
        x = x / 2
        y = y / 2
        depth += 1
    return depth


# PYRAMID CALCULATIONS

# Gaussian Pyramid
def gaussian_pyramid(image, level, sigma=1.1):
    """

    :param image: grayscale image to use
    :param level: the depth of the pyramid
    :param sigma: the value of sigma of gaussian filter. default is 1.1
    :return: gaussian pyramid as list
    """
    gauss_pyramid = [image]
    for k in range(1, level):
        image2 = np.zeros(image.shape)
        image2 = gaussian_filter(image, sigma=sigma) #gaussian blur
        image2 = image2[::2, ::2]  # down-sampling
        image = image2
        gauss_pyramid.append(image2)
    return gauss_pyramid


# Gaussian Pyramid Color
def gaussian_pyramid_color(image, level, sigma=1.1):
    """
    colored version of gaussian_pyramid
    :param image: RGB image to use
    :param level: the depth of the pyramid
    :param sigma: the value of sigma of gaussian filter. default is 1.1
    :return: gaussian pyramid as list
    """
    gauss_pyramid = [image]
    for k in range(1, level):
        image2 = np.zeros(image.shape)
        for z in range(3):  # because image in color
            image2[:, :, z] = gaussian_filter(image[:, :, z], sigma=sigma)
        image2 = image2[::2, ::2, :]  # down-sampling
        image = image2
        gauss_pyramid.append(image2)
    return gauss_pyramid


# Laplacian Pyramid
def laplacian_pyramid(gauss_pyramid, level):
    """
    creates a laplacian pyramid
    :param gauss_pyramid: the gauss pyramid of the wanted image
    :param level: depth of pyramid
    :return: laplacian pyramid as list
    """
    lap_pyramid = []
    for k in range(0, level - 1):  # pyramid levels
        lvl1 = gauss_pyramid[k]
        lvl2 = gauss_pyramid[k + 1]
        # downsampling with lvl1 shape to match pixels
        lvl2 = cv2.resize(lvl2, (lvl1.shape[1], lvl1.shape[0]))
        image = lvl1 - lvl2
        lap_pyramid.append(image)
    lap_pyramid.append(gauss_pyramid[level - 1])
    return lap_pyramid


# Collapsing the pyramid
def reconstructing_pyramid(pyramid):
    """
    collapsing the pyramid.
    :param pyramid: pyramid as list to reconstruct
    :return: the image which pyramid belonged
    """
    reverse_pyramid = pyramid[::-1]
    image = reverse_pyramid[0]
    for i in range(1, len(reverse_pyramid)):
        image = cv2.resize(image, (reverse_pyramid[i].shape[1], reverse_pyramid[i].shape[0])) + reverse_pyramid[i]
    return image


# Calculation the blended image
def blend_pyramids(pyramid_white, pyramid_black, pyramid_mask):
    """
    Blend two images by applying the Laplacian pyramid blending technique. Using the formula in pdf.
    :param pyramid_white: the original image pyramid
    :param pyramid_black: the image pyramid we want to blend to the original image
    :param pyramid_mask: the mask to perform on pyramids
    :return: the blended image pyramid
    """
    blended_img = []
    for i in range(len(pyramid_black)):
        # dimension up to calculate in RGB
        pyramid_mask[i] = cv2.merge([pyramid_mask[i]] * 3)
        # normalization
        pyramid_mask[i] = pyramid_mask[i] / 255.0
        blended = ((pyramid_mask[i]) * pyramid_white[i]) + ((1 - pyramid_mask[i]) * pyramid_black[i])
        blended_img.append(blended / 255.0)
    return blended_img


# Displaying the pyramid
def displaying_pyramid(pyramid):
    level = len(pyramid)
    fig, ax = plt.subplots(nrows=1, ncols=level, figsize=(15, 7), dpi=72,
                           sharex=True, sharey=True)
    for k in range(level - 1, -1, -1):
        # Ensure pixel values are in the range [0, 1]
        img_to_show = pyramid[k] - np.min(pyramid[k])
        img_to_show = img_to_show / np.max(img_to_show)
        ax[k].imshow(img_to_show)
    plt.show()


# MASK CALCULATIONS

# creating a mask left white right black
def create_half_mask_for(image):
    mask = np.zeros((image.shape[0], image.shape[1]))
    ncols = image.shape[1]
    mask[:, :int(ncols / 2)] = 255
    return mask

# creating a mask up white down black
def create_horiz_half_mask_for(image):
    mask = np.zeros((image.shape[0], image.shape[1]))
    nrows = mask.shape[0]
    mask[:int(nrows / 2), :] = 255
    return mask


# creating a mask down diagonal white up diagonal black
def create_diag_mask_for(image):
    mask = np.zeros((image.shape[0], image.shape[1]))
    nrows, ncols = image.shape[0:2]
    for i in range(nrows):
        for j in range(min(i, ncols)):
            mask[i, j] = 255
    return mask


def select_mask_region(source_image):
    # selecting ROI
    roi_mask = cv2.selectROI('Select Mask Space', source_image, fromCenter=False, showCrosshair=True)
    # create a binary mask
    mask = np.full((source_image.shape[0], source_image.shape[1]), 255)
    # setting pixels inside the ROI to 255
    mask[int(roi_mask[1]):int(roi_mask[1] + roi_mask[3]), int(roi_mask[0]):int(roi_mask[0] + roi_mask[2])] = 0
    return [mask, roi_mask]


def fit_image_to_mask(destination_image, fitted_image, roi_mask, padded_white):
    mask_area_x = roi_mask[2]
    mask_area_y = roi_mask[3]

    # resizing image to fit the mask perfectly
    selected_image = cv2.resize(fitted_image, (mask_area_x, mask_area_y))
    if padded_white:
        padded_image = np.full((destination_image.shape[0], destination_image.shape[1], 3), 255,dtype=np.uint8)
    else:
        padded_image = np.full((destination_image.shape[0], destination_image.shape[1], 3), selected_image[0][0], dtype=np.uint8)

    # dealing with indexes
    roi_top = roi_mask[1]
    roi_left = roi_mask[0]
    roi_bottom = roi_top + selected_image.shape[0]
    roi_right = roi_left + selected_image.shape[1]

    # padding the image with ones according to given mask
    padded_image[roi_top:roi_bottom, roi_left:roi_right] = selected_image
    # plt.imshow(padded_image)
    # plt.show()
    return padded_image


# blends two images by taking the mask as roi input
# and can use one or two img
def blend_image_mode1(img1, img2, level=None, display=False, padded_white=True):
    if level is None:
        level = find_pyramid_depth(img1)

    img2 = select_image_from(img2)

    mask, roi_mask = select_mask_region(img1)

    if display:
        plt.imshow(mask, cmap="gray")
        plt.show()

    # to save mask
    # plt.imsave('luffy_mask.jpg', mask,cmap="gray")

    img2 = fit_image_to_mask(img1, img2, roi_mask,padded_white)

    # to save clipped image
    # plt.imsave('luffy_clipped.jpg', img2)

    return blend_image_mode2(img1, img2, mask, level, display)


# blends two images by taking the mask as input
# and uses two separate images
def blend_image_mode2(img1, img2, mask, level=None, display=False):
    if level is None:
        depth = find_pyramid_depth(img1)
    else:
        depth = level

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # pyramid for first image
    G1 = gaussian_pyramid_color(img1, depth, 1.2)
    L1 = laplacian_pyramid(G1, depth)

    # pyramid for second image
    G2 = gaussian_pyramid_color(img2, depth, 1.2)
    L2 = laplacian_pyramid(G2, depth)

    GM = gaussian_pyramid(mask, depth, 2)

    result_pyramid = blend_pyramids(L1, L2, GM)

    if display:
        displaying_pyramid(G1)
        displaying_pyramid(L1)
        displaying_pyramid(G2)
        displaying_pyramid(L2)
        displaying_pyramid(GM)
        displaying_pyramid(result_pyramid)

    result_image = reconstructing_pyramid(result_pyramid)
    result_image = np.clip(result_image, 0, 1)
    plt.imshow(result_image)
    plt.show()

    return result_image


img_org = plt.imread('miyeon1_clipped.jpg')
img_other = plt.imread('miyeon2_clipped.jpg')
pyramid_level = 10
pyramid_mask = create_half_mask_for(img_org)
last_image = blend_image_mode2(img_org, img_other, pyramid_mask, level=pyramid_level)
# for saving the result image
# plt.imsave(f'output_image_{pyramid_level}.jpg', last_image)

