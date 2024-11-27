import base64
import math
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Remove limitation on max pixels per image
Image.MAX_IMAGE_PIXELS = None


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
    # Create the SIFT feature detector
    sift = cv2.SIFT_create()
    # Detect keypoints of important features and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def find_matches(descriptors1, descriptors2):
    """Finds Matches Between Descriptors of Images Using Brute Force Matching"""
    # Create the Brute Force feature matcher
    bf_matcher = cv2.BFMatcher()
    # Find matches between the descriptors of the two images
    matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)
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
    biggest_contour = max(contours, key=cv2.contourArea)
    # Find the top-left point of the bounding rectangle
    top_left = np.squeeze(biggest_contour.min(axis=0)).tolist()
    # Find the bottom-right point of the bounding rectangle
    bottom_right = np.squeeze(biggest_contour.max(axis=0)).tolist()
    return top_left, bottom_right


def stitch_images(base64_image1, base64_image2):
    """
    Stitches two images together by aligning them based on matched keypoints and descriptors.
    Params:
        base64_image1: The first image encoded to a base64 string.
        base64_image2: The second image encoded to a base64 string.
    Returns:
        tuple:
            If stitching is successful:
                (True, base64 string of stitched image)
            If stitching fails:
                (False, failure reason, status code)
    """
    if not isinstance(base64_image1, str) or not isinstance(base64_image2, str):
        return (False, "קלט אינו תקין!", 400)

    action = ""
    try:
        # Decode the images from base64 strings to numpy arrays
        action = "שינוי פורמט התמונות שהתקבלו"
        image1 = decode_image(base64_image1)
        image2 = decode_image(base64_image2)

        # Get the dimensions of the images
        action = "קבלת אורך ורוחב התמונות"
        image1_height, image1_width = image1.shape[:2]
        image2_height, image2_width = image2.shape[:2]

        # Set scaling factors for downscaling the images to improve performance
        action = "חישוב מקדם הקטנה לכל תמונה"
        scaling_factor1 = min(image1_height, image1_width) / 1000 if min(image1_height, image1_width) > 1000 else 1
        scaling_factor2 = min(image2_height, image2_width) / 1000 if min(image2_height, image2_width) > 1000 else 1

        # Calculate the new dimensions for both images based on their scaling factors
        action = "חישוב אורך ורוחב הגרסאות המוקטנות של התמונות"
        downscaled_image1_width = int(image1_width / scaling_factor1)
        downscaled_image1_height = int(image1_height / scaling_factor1)
        downscaled_image2_width = int(image2_width / scaling_factor2)
        downscaled_image2_height = int(image2_height / scaling_factor2)

        # Downscale the images to make feature extraction and feature matching faster and reduce memory usage
        action = "הקטנת התמונות"
        downscaled_image1 = cv2.resize(image1, (downscaled_image1_width, downscaled_image1_height))
        downscaled_image2 = cv2.resize(image2, (downscaled_image2_width, downscaled_image2_height))

        # Extract keypoints and descriptors from the downscaled images
        action = "חילוץ נקודות חשובות מהתמונות"
        downscaled_keypoints1, downscaled_descriptors1 = extract_keypoints_and_descriptors(downscaled_image1)
        downscaled_keypoints2, downscaled_descriptors2 = extract_keypoints_and_descriptors(downscaled_image2)

        # Find matches between the descriptors of the downscaled images using brute force
        action = "חיפוש התאמות בין התמונות"
        matches = find_matches(downscaled_descriptors1, downscaled_descriptors2)

        # Get the strong matches based on Lowe's ratio test
        action = "הסרת התאמות חלשות"
        strong_matches = get_strong_matches(matches)

        # Check if the amount of strong matches between the images is not enough for good stitching
        action = "בדיקת כמות ההתאמות הטובות בין התמונות"
        if len(strong_matches) < 25:
            return (False, "אין חפיפה בין התמונות שנבחרו!", 500)

        # Upscale the keypoints' coordinates back to match the original image sizes
        action = "הגדלת הקואורדינטות של הנקודות ביניהן ישנן התאמות"
        keypoints1 = [cv2.KeyPoint(scaling_factor1 * keypoint.pt[0], scaling_factor1 * keypoint.pt[1], 1) for keypoint in downscaled_keypoints1]
        keypoints2 = [cv2.KeyPoint(scaling_factor2 * keypoint.pt[0], scaling_factor2 * keypoint.pt[1], 1) for keypoint in downscaled_keypoints2]

        # Compute the homography matrix that aligns the first image to the second image's perspective
        action = "חישוב מטריצת ההמרה בין פרספקטיבות"
        homography_matrix = compute_homography(strong_matches, keypoints1, keypoints2)

        # Compute the warped corners of the first image based on the homography matrix
        action = "חישוב קואורדינטות פינות התמונה הראשונה לאחר שינוי הפרספקטיבה"
        image1_corners = np.array([(0, 0), (image1_width, 0), (image1_width, image1_height), (0, image1_height)], dtype=np.float32).reshape((4, 1, 2))
        warped_image1_corners = cv2.perspectiveTransform(image1_corners, homography_matrix).reshape((4, 2))

        # The top-left and bottom-right points of the rectangle bounding the warped first image
        top_left = np.min(warped_image1_corners, axis=0)
        bottom_right = np.max(warped_image1_corners, axis=0)

        # Calculate the padding needed so the entire warped first image fits within the second image
        action = "חישוב הריפוד הדרוש עבור התאמת התמונה הראשונה"
        top_pad = -top_left[1] if top_left[1] < 0 else 0
        left_pad = -top_left[0] if top_left[0] < 0 else 0
        bottom_pad = bottom_right[1] - image2_height if bottom_right[1] > image2_height else 0
        right_pad = bottom_right[0] - image2_width if bottom_right[0] > image2_width else 0

        # Round up the padding values for practical use in cv2.copyMakeBorder
        action = "עיגול ערכי הריפוד"
        left_pad = math.ceil(left_pad)
        top_pad = math.ceil(top_pad)
        bottom_pad = math.ceil(bottom_pad)
        right_pad = math.ceil(right_pad)

        # Pad the second image to ensure the first image can be safely stitched onto it without exceeding the borders and losing data
        action = "ריפוד התמונה השנייה כך שתוכל להכיל את כל התמונה הראשונה"
        padded_image2 = cv2.copyMakeBorder(image2, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT)
        padded_image2_height, padded_image2_width = padded_image2.shape[:2]

        # Define a translation matrix for the required padding
        action = "הגדרת מטריצת תזוזה עבור הריפוד הנדרש"
        translation_matrix = np.array([[1, 0, left_pad],
                                       [0, 1, top_pad],
                                       [0, 0, 1]])

        # Combine the homography with the translation matrix to align image1 with padded image2
        action = "שילוב מטריצת ההמרה בין פרספקטיבות ומטריצת ההזזה למטריצה משולבת שמבצעת את שתי הפעולות"
        combined_transformation_matrix = translation_matrix @ homography_matrix  # homography performed first and then the translation

        # Warp the first image using the combined transformation matrix
        action = "עיוות התמונה הראשונה באמצעות מטריצת ההמרה המשולבת"
        warped_image1 = cv2.warpPerspective(image1, combined_transformation_matrix, (padded_image2_width, padded_image2_height))

        # Create a copy of the padded second image that will hold the result of the stitching between the first and second images
        stitched_image = padded_image2.copy()

        # Create a binary mask for non-padding pixels in the warped first image (the parts of the image that will used in the stitching)
        action = "יצירת מסכה בינארית עבור אזורים שאינם ריפוד בתמונה המעוותת"
        warped_image1_not_padding = warped_image1 != 0

        # Replace the corresponding region of the padded second image with the warped first image
        action = "החלת המסכה לשילוב התמונות"
        stitched_image[warped_image1_not_padding] = warped_image1[warped_image1_not_padding]

        # Find the smallest bounding rect that can bound the non-padding area of the stitched image
        bounding_rect_top_left, bounding_rect_bottom_right = find_smallest_bounding_rect(stitched_image)

        # Crop the stitched image to remove padding from its sides
        cropped_stitched_image = stitched_image[bounding_rect_top_left[1]: bounding_rect_bottom_right[1], bounding_rect_top_left[0]: bounding_rect_bottom_right[0]]

        # Encode the stitched image to base64
        action = "שינוי פורמט התמונה המחוברת"
        base64_stitched_image = encode_image(cropped_stitched_image)

        return (True, base64_stitched_image)

    except Exception:
        return (False, f"התרחשה שגיאה בלתי צפויה במהלך {action}", 500)


def main():
    # Reading the images
    image1 = plt.imread("images/1.jpg")
    image2 = plt.imread("images/2.jpg")

    # Encoding the images to base64 strings
    encoded_image1 = encode_image(image1)
    encoded_image2 = encode_image(image2)

    # Performing stitching between the images
    stitching_result = stitch_images(encoded_image1, encoded_image2)

    # Get the status (success or failure) of the stitching operation
    is_stitching_successful = stitching_result[0]

    # Check if the stitching was successful
    if is_stitching_successful:
        # Get the base64 encoded stitching images
        base64_stitched_image = stitching_result[1]

        # Decode the base64 representation of the stitched image
        stitched_image = decode_image(base64_stitched_image)

        # Display the stitching result
        plt.figure(figsize=(7, 8))
        plt.subplot(321)
        plt.title("First image")
        plt.imshow(image1)
        plt.subplot(322)
        plt.title("Second image")
        plt.imshow(image2)
        plt.subplot(3, 2, (3, 6))
        plt.title("Stitched image")
        plt.imshow(stitched_image)
        plt.tight_layout()
        plt.show()

    else:
        # Get the reason that caused the stitching to fail
        stitching_failure_reason = stitching_result[1]
        print('Stitching failed!')
        print(stitching_failure_reason)


if __name__ == "__main__":
    main()
