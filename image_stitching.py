import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image(image_path):
    """Loads Image from Path"""
    # Read the image from the path
    image_bgr = cv2.imread(image_path)
    # Convert the color-space of the image from BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def extract_keypoints_and_descriptors(image):
    """Extracts Keypoints and Descriptors from Image"""
    # Convert the color-space of the image from RGB to Grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Create the SIFT feature detector
    sift = cv2.SIFT_create()
    # Detect keypoints of important features and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    return keypoints, descriptors


def find_matches(descriptors1, descriptors2):
    """Finds Matches Between Descriptors of Images Using FLANN"""
    # Configure the parameters of the FLANN feature matcher
    FLANN_INDEX_KDTREE = 0
    index_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5}
    search_params = {"checks": 50}
    # Create the FLANN feature matcher
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    # Find matches between the descriptors of the two images
    matches = flann_matcher.knnMatch(descriptors1, descriptors2, k=2)
    return matches


def get_strong_matches(matches):
    """Keeps Only the Strong Matches Based on Lowe's Ratio Test"""
    # For each pair of matches, check the distance between the first and second match,
    # If the first is significantly better, keep it.
    best_matches = [[match1] for match1, match2 in matches if match1.distance < 0.6 * match2.distance]
    return best_matches


def display_matches(image1, keypoints1, image2, keypoints2, matches, match_color=(0, 255, 0), single_point_Color=(255, 0, 0), matches_mask=None):
    """Displays Matched Keypoints Between Images"""
    # Check whether a matches mask was provided as a parameter
    if matches_mask:
        # If a mask was provided, use the default flags to draw the matches
        flags = cv2.DRAW_MATCHES_FLAGS_DEFAULT
    else:
        # If no mask was provided, do not draw single unmatched points
        flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS

    # Draw the matches between the keypoints of both images
    matches_image = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None, match_color, single_point_Color, matches_mask, flags)

    # Display the image with matches using matplotlib
    plt.figure(figsize=(12, 6))
    plt.imshow(matches_image)
    plt.title("Feature Matching with SIFT (Scale-Invariant Feature Transform) and FLANN Matcher")
    plt.tight_layout()
    plt.show()


def compute_homography(matches, keypoints1, keypoints2):
    """Computes Homography Matrix"""
    # Create an array that contains the coordinates (x, y) of the features from the first image
    points1 = np.array([keypoints1[match[0].queryIdx].pt for match in matches], dtype=np.float32)
    # Create an array that contains the coordinates (x, y) of the features from the second image
    points2 = np.array([keypoints2[match[0].trainIdx].pt for match in matches], dtype=np.float32)
    # Find a homography matrix that brings the coordinates of each point from the first array to the coordinates of the second array while ignoring outliers
    homography_matrix, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    return homography_matrix


def find_smallest_bounding_rect(image):
    """Finds the Top-Left and Bottom-Right Coordinates of the Smallest Rectangle that Can Bound the non-black Pixels of the Image"""
    # Convert the color-space of the image from RGB to Grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Create a binary mask of the image by setting all non-black pixels to 255, and black (padding) to 0
    _, image_mask = cv2.threshold(src=image_gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
    # Find the contours of the mask
    contours, _ = cv2.findContours(image=image_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # Keep the biggest contour of the mask (if for some reason it has more than one contour)
    biggest_contour = sorted(list(contours), key=cv2.contourArea, reverse=True)[0]
    # Find the top-left point of the bounding rectangle
    top_left = np.squeeze(biggest_contour.min(axis=0)).tolist()
    # Find the bottom-right point of the bounding rectangle
    bottom_right = np.squeeze(biggest_contour.max(axis=0)).tolist()
    return top_left, bottom_right


def refine_contour(contour, epsilon_factor=0.01):
    """
    Refines the contour by reducing the number of points using contour approximation.
    Params:
        contour: The input contour to refine.
        epsilon_factor: A factor that determines the approximation accuracy. Higher value = more reduction.
    Returns:
        refined_contour: The refined contour with fewer points.
    """
    # Calculate the perimeter of the contour
    contour_perimeter = cv2.arcLength(contour, closed=True)
    # Epsilon determines the approximation accuracy.
    epsilon = epsilon_factor * contour_perimeter
    # Approximate the contour with fewer points
    refined_contour = cv2.approxPolyDP(contour, epsilon, True)
    return refined_contour


def main():
    # Load the images
    image1 = load_image("images/23.jpg")
    image2 = load_image("images/25.jpg")

    # Get the dimensions of the images
    image1_height, image1_width = image1.shape[:2]
    image2_height, image2_width = image2.shape[:2]

    # Pad the second image to ensure the first image can be stitched onto it without exceeding the borders
    vertical_pad = image1_height
    horizontal_pad = image1_width
    padded_image2 = cv2.copyMakeBorder(image2, vertical_pad, vertical_pad, horizontal_pad, horizontal_pad, cv2.BORDER_CONSTANT)

    # Get the dimensions of the padded second image
    padded_image2_height, padded_image2_width = padded_image2.shape[:2]

    # Set scaling factors for downscaling the images to improve performance
    # Ensures that images are downscaled as much as possible while keeping their smaller dimension above 1000
    scaling_factor1 = min(image1_height, image1_width) / 1000 if min(image1_height, image1_width) > 1000 else 1
    scaling_factor2 = min(image2_height, image2_width) / 1000 if min(image2_height, image2_width) > 1000 else 1

    # Calculate the new dimensions for both images based on their scaling factors
    downscaled_image1_width = int(image1_width / scaling_factor1)
    downscaled_image1_height = int(image1_height / scaling_factor1)
    downscaled_padded_image2_width = int(padded_image2_width / scaling_factor2)
    downscaled_padded_image2_height = int(padded_image2_height / scaling_factor2)

    # Downscale the images to make feature extraction and feature matching faster and reduce memory usage
    downscaled_image1 = cv2.resize(image1, (downscaled_image1_width, downscaled_image1_height))
    downscaled_padded_image2 = cv2.resize(padded_image2, (downscaled_padded_image2_width, downscaled_padded_image2_height))

    # Extract keypoints and descriptors from the downscaled images
    downscaled_keypoints1, downscaled_descriptors1 = extract_keypoints_and_descriptors(downscaled_image1)
    downscaled_keypoints2, downscaled_descriptors2 = extract_keypoints_and_descriptors(downscaled_padded_image2)

    # Find matches between the descriptors of the downscaled images using FLANN
    matches = find_matches(downscaled_descriptors1, downscaled_descriptors2)

    # Get the strong matches based on Lowe's ratio test
    strong_matches = get_strong_matches(matches)

    # Upscale the keypoints' coordinates back to match the original image sizes
    keypoints1 = [cv2.KeyPoint(scaling_factor1 * keypoint.pt[0], scaling_factor1 * keypoint.pt[1], 1) for keypoint in downscaled_keypoints1]
    keypoints2 = [cv2.KeyPoint(scaling_factor2 * keypoint.pt[0], scaling_factor2 * keypoint.pt[1], 1) for keypoint in downscaled_keypoints2]

    # Compute the homography matrix that aligns the first image to the padded second image's perspective
    homography_matrix = compute_homography(strong_matches, keypoints1, keypoints2)

    # Warp the first image using the computed homography to match the second image's perspective
    warped_image1 = cv2.warpPerspective(image1, homography_matrix, (padded_image2_width, padded_image2_height))

    # Create a copy of the padded second image that will hold the result of the stitching between the first and second images
    stitched_image = padded_image2.copy()

    # Create a binary mask indicating what parts of the first image should be copied into the stitched image
    warped_image1_not_padding = warped_image1 != 0

    # Replace the corresponding region of the padded second image with the aligned first image
    stitched_image[warped_image1_not_padding] = warped_image1[warped_image1_not_padding]

    # Find the smallest rectangular region in the stitched image that contains all the non-padding pixels
    top_left, bottom_right = find_smallest_bounding_rect(stitched_image)

    # Crop the smallest bounding rect that contains all the non-padding pixels
    cropped_stitched_image = stitched_image[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]

    # Display the original images and the stitched image
    plt.figure(figsize=(10, 8))
    plt.subplot(231)
    plt.imshow(image1)
    plt.title("Original First Image")
    plt.subplot(234)
    plt.imshow(image2)
    plt.title("Original Second Image")
    plt.subplot(2, 3, (2, 6))
    plt.imshow(cropped_stitched_image)
    plt.title("Stitched Image")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()