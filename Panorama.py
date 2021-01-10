import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

# arguments to take from console
parser = argparse.ArgumentParser(
    description="Making a panorama picture using a left and right picture to create it ...",
    epilog="And that's it :) ... ")
parser.add_argument('left_p', type=str, metavar='<left picture path>',
                    help="please enter a picture path to be on the left side of the panorama")
parser.add_argument('right_p', type=str, metavar='<right picture path>',
                    help="please enter a picture path to be on the right side of the panorama")
parser.add_argument('final_p', type=str, metavar='<the path you want to save the panorama at>',
                    help="please enter a path for the panorama to be saved at or just the"
                         " name of the file and it will be saved"
                         ".")
args = parser.parse_args()


def resize_image_by_height(left_img, right_img):
    """
    we resize the images while keeping the aspect ration intact
    :param left_img: the image we want to resize
    :param right_img: the image we use it's height to resize
    :return: return the left image resized
    the left and image is just for making the names easy, but left can be right and vice versa
    """
    left_height, left_width, _ = left_img.shape
    desired_height, _, _ = right_img.shape

    # taking the ratio of the original size of left image
    ratio = left_height / left_width

    # create new width using desire height and old ratio
    new_w = int(desired_height / ratio)

    # return the image with the new dims
    return cv2.resize(left_img, (new_w, desired_height))


def resize_image(img):
    """
    :arg img: an image with big dimensions
    :returns:resized image
    """
    return cv2.resize(img, (0, 0), fx=0.7, fy=0.7)


def from_bgr_to_gray(img):
    """
    :arg img:an image we want to turn from bgr to grayscale
    :returns: a grayscale image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def trim(frame):
    # crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    # crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    # crop top
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    # crop top
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


def make_panorama(left_path, right_path, panorama_path, draw_match=False):
    """
    :param draw_match: if we want to to draw the matches after we detect them on the two pictures
    :arg left_path: an image path to be put on the left side of the panorama
    :arg right_path: an image path to put on the right side of the panorama
    :arg panorama_path: a path to save the panorama (which made of the two pictures)   
    """
    leftpic = cv2.imread(left_path)
    rightpic = cv2.imread(right_path)

    left_height, w_l, _ = leftpic.shape  # we take the height
    right_height, w_r, _ = rightpic.shape  # we take the height to check which picture has the biggest height

    # resizing big images
    if left_height * w_l > 1000000:
        leftpic = resize_image(leftpic)
        left_height, _, _ = leftpic.shape
    if right_height * w_r > 1000000:
        rightpic = resize_image(rightpic)
        right_height, _, _ = rightpic.shape

    # checking the height of each image to know what to resize
    if left_height > right_height:
        leftpic = resize_image_by_height(leftpic, rightpic)
    elif left_height < right_height:
        rightpic = resize_image_by_height(rightpic, leftpic)

    # we need the images in gray scale in order for the sift algorithm to work
    left_gray_pic = from_bgr_to_gray(leftpic)
    right_gray_pic = from_bgr_to_gray(rightpic)

    # using the sift algorithm
    sift = cv2.xfeatures2d.SIFT_create()
    keypointsLeft, descriptorsLeft = sift.detectAndCompute(left_gray_pic, None)
    keypointsRight, descriptorsRight = sift.detectAndCompute(right_gray_pic, None)

    bf = cv2.BFMatcher()

    raw_matches = bf.knnMatch(descriptorsRight, descriptorsLeft, k=2)

    # the ratio of distance because we get two neighbors
    ratio = 0.85

    good_matches = []

    for m1, m2 in raw_matches:
        if m1.distance < ratio * m2.distance:
            # m1 is a good match, save it
            good_matches.append([m1])

    left_height, left_width = left_gray_pic.shape  # we take the height
    right_height, right_width = right_gray_pic.shape
    if draw_match:
        img_match = np.empty((max(left_height, right_height), left_width + right_width, 3),
                             dtype=np.uint8)
        imMatches = cv2.drawMatchesKnn(leftpic, keypointsLeft, rightpic, keypointsRight, good_matches, img_match, None)
        cv2.imshow('Matches between the two pictures', imMatches)

    # taking the key points of the two pictures
    right_image_kp = np.float32([keypointsRight[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    left_image_kp = np.float32([keypointsLeft[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # finding the homography matrix
    H, status = cv2.findHomography(right_image_kp, left_image_kp, cv2.RANSAC, 5.0)

    width_panorama = left_width + right_width
    height_panorama = right_height

    res_pa = cv2.warpPerspective(rightpic, H, (width_panorama, height_panorama))

    res_pa[0:leftpic.shape[0], 0:leftpic.shape[1]] = leftpic

    res_pa = trim(res_pa)
    cv2.imwrite(panorama_path, res_pa)
    print("Panorama saved in given path")


if __name__ == '__main__':
    if args.left_p and args.right_p and args.final_p:
        make_panorama(args.left_p, args.right_p, args.final_p)
    else:
        print("In order to run the script you need to enter three arguments , use --help for more information")
    # make_panorama("left.jpg", "right.jpg", "pano.jpg")
