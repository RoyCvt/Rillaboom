import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def decode_image(base64_string):
    """Decode an Image from Base64 String to Numpy Array"""
    # Decode the base64 string to binary
    image_data = base64.b64decode(base64_string)
    # Convert binary data to an image using BytesIO
    image = Image.open(BytesIO(image_data))
    # Convert the PIL image to a NumPy array
    np_image = np.array(image)
    return np_image


def encode_image(np_image):
    """Encode an Image from Numpy Array to Base64 String"""
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(np_image)
    # Save the image to an in-memory buffer (BytesIO)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    # Encode the image data to base64
    base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_string


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


def find_biggest_bounded_rect(image):
    """Finds the Top-Left and Bottom-Right Coordinates of the Biggest Rectangle that Can Be Bound by the non-black Pixels of the Image"""
    # Convert the color-space of the image from RGB to Grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Create a binary mask of the image by setting all non-black pixels to 255, and black (padding) to 0
    _, image_mask = cv2.threshold(src=image_gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
    # Find the contours of the mask
    contours, _ = cv2.findContours(image=image_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # Keep the biggest contour of the mask (if for some reason it has more than one contour)
    biggest_contour = sorted(list(contours), key=cv2.contourArea, reverse=True)[0]
    # Refine the contour by getting rid of redundant points
    biggest_contour = refine_contour(biggest_contour)
    # Get rid of redundant dimensions in the array
    biggest_contour = np.squeeze(biggest_contour)
    # Find the centroid of the contour using moments
    m = cv2.moments(biggest_contour)
    cx = int(m["m10"] / m["m00"])  # x-coordinate of the contour's center
    cy = int(m["m01"] / m["m00"])  # y-coordinate of the contour's center
    # Separate the x and y values of the contour points
    x_values = biggest_contour[:, 0]
    y_values = biggest_contour[:, 1]
    # Get the x-coordinate that is positioned left to the centroid's x (cx)
    left = max(x for x in x_values if x < cx)
    # Get the left-most x-coordinate that is positioned right to the centroid's x (cx)
    right = min(x for x in x_values if x > cx)
    # Get the down-most y-coordinate that is positioned above the centroid's y (cy)
    top = max(y for y in y_values if y < cy)
    # Get the up-most y-coordinate that is positioned below the centroid's y (cy)
    bottom = min(y for y in y_values if y > cy)
    # Construct the top-left point of the bounded rectangle
    top_left = (left, top)
    # Construct the bottom-right point of the bounded rectangle
    bottom_right = (right, bottom)
    return top_left, bottom_right


def register_images(base64_image1, base64_image2):
    """
    Performs registration between 2 images.
    Params:
        base64_image1: The first image encoded to a string with base64.
        base64_image2: The second image encoded to a string with base64.
    Returns:
        tuple:
            If registration is successful:
                (True, first registered image encoded to base64 string, second registered image encoded to base64 string)
            If registration fails:
                (False, failure reason, status code)
    """
    if not isinstance(base64_image1, str) or not isinstance(base64_image2, str):
        return (False, 'Bad input', 400)

    action = ""
    try:
        # Decode the images from base64 strings to numpy arrays
        action = 'decoding the images from base64 strings to numpy arrays'
        image1 = decode_image(base64_image1)
        image2 = decode_image(base64_image2)

        # Get the dimensions of the images
        action = 'getting the height and width of the images'
        image1_height, image1_width = image1.shape[:2]
        image2_height, image2_width = image2.shape[:2]

        # Set scaling factors for downscaling the images to improve performance
        # Ensures that images are downscaled as much as possible while keeping their smaller dimension above 1000
        action = 'calculating the optimal scaling factor for each of the images'
        scaling_factor1 = min(image1_height, image1_width) / 1000 if min(image1_height, image1_width) > 1000 else 1
        scaling_factor2 = min(image2_height, image2_width) / 1000 if min(image2_height, image2_width) > 1000 else 1

        # Calculate the new dimensions for both images based on their scaling factors
        action = 'calculating the width and height of the downscaled versions of the images'
        downscaled_image1_width = int(image1_width / scaling_factor1)
        downscaled_image1_height = int(image1_height / scaling_factor1)
        downscaled_image2_width = int(image2_width / scaling_factor2)
        downscaled_image2_height = int(image2_height / scaling_factor2)

        # Downscale the images to make feature extraction and feature matching faster and reduce memory usage
        action = 'downscaling the images'
        downscaled_image1 = cv2.resize(image1, (downscaled_image1_width, downscaled_image1_height))
        downscaled_image2 = cv2.resize(image2, (downscaled_image2_width, downscaled_image2_height))

        # Extract keypoints and descriptors from the downscaled images
        action = "extracting keypoints and descriptors from the images"
        downscaled_keypoints1, downscaled_descriptors1 = extract_keypoints_and_descriptors(downscaled_image1)
        downscaled_keypoints2, downscaled_descriptors2 = extract_keypoints_and_descriptors(downscaled_image2)

        # Find matches between the descriptors of the downscaled images using FLANN
        action = "searching for matches between the images"
        matches = find_matches(downscaled_descriptors1, downscaled_descriptors2)

        # Get the strong matches based on Lowe's ratio test
        action = "removing weak matches"
        strong_matches = get_strong_matches(matches)

        # Upscale the keypoints' coordinates back to match the original image sizes
        action = "upsacling the coordinates of the matches' keypoints"
        keypoints1 = [cv2.KeyPoint(scaling_factor1 * keypoint.pt[0], scaling_factor1 * keypoint.pt[1], 1) for keypoint in downscaled_keypoints1]
        keypoints2 = [cv2.KeyPoint(scaling_factor2 * keypoint.pt[0], scaling_factor2 * keypoint.pt[1], 1) for keypoint in downscaled_keypoints2]

        # Compute the homography matrix that aligns the first image to the second image's perspective
        action = "computing the homography matrix"
        homography_matrix = compute_homography(strong_matches, keypoints1, keypoints2)

        # Warp the first image using the computed homography to match the second image's perspective
        action = "warping the perspective of the first image based on the homography matrix"
        warped_image1 = cv2.warpPerspective(image1, homography_matrix, (image2_width, image2_height))

        # Find the biggest rectangular region in the warped first image that doesn't contain any padding (black areas)
        action = "finding the largest area without padding in the first image"
        top_left, bottom_right = find_biggest_bounded_rect(warped_image1)

        # Crop the common region between the two images to prepare for overlay
        action = "cropping the images in the same places so they become registered and without padding"
        cropped_warped_image1 = warped_image1[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
        cropped_image2 = image2[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]

        # Encode the images to base64
        action = "encoding the registration images from numpy arrays to base64 strings"
        base64_registered_image1 = encode_image(cropped_warped_image1)
        base64_registered_image2 = encode_image(cropped_image2)

        return (True, base64_registered_image1, base64_registered_image2)

    except Exception:
        return (False, f"Encountered unexpected error while {action}", 500)
