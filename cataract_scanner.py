import cv2
import numpy as np
import sys


def read_image(filepath: str):
    """
    Read the image in both color and grayscale formats.

    Parameters:
    - filepath (str): The path to the image file.

    Returns:
    - tuple: A tuple containing the BGR and grayscale images.
    """
    img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img_bgr, img_gray


def resize_and_preprocess(img_gray, img_bgr):
    """
    Resize the image based on its dimensions and apply preprocessing.

    Parameters:
    - img_gray: The grayscale image.
    - img_bgr: The BGR image.

    Returns:
    - tuple: A tuple containing the processed BGR and grayscale images, and the Hough factor.
    """
    height, width = img_gray.shape
    if height > 900:
        resized_dim = (width // 10, height // 10)
        factor = 300
    elif height < 200:
        resized_dim = (width * 2, height * 2)
        factor = 300
    else:
        resized_dim = (width, height)
        factor = 500

    img_gray = cv2.resize(img_gray, resized_dim, interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.medianBlur(img_gray, 5)
    img_bgr = cv2.resize(img_bgr, resized_dim, interpolation=cv2.INTER_CUBIC)
    return img_bgr, img_gray, factor


def detect_and_draw_circles(img_gray, img_bgr, factor):
    """
    Detect circles using HoughCircles and draw them on the BGR image.

    Parameters:
    - img_gray: The grayscale image.
    - img_bgr: The BGR image.
    - factor: The factor for the HoughCircles method.

    Returns:
    - ndarray or None: Detected circles or None.
    """
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, factor, param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0]:
            # Draw circles
            cv2.circle(img_bgr, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.circle(img_bgr, (circle[0], circle[1]), 2, (0, 0, 255), 3)
    return circles


def evaluate_cataract(img_gray, circles):
    """
    Evaluate whether the detected circle region has cataract or not.

    Parameters:
    - img_gray: The grayscale image.
    - circles: Detected circles.

    Returns:
    - str: The evaluation message.
    """
    xc, yc, r = circles[0][0]
    y, x = np.ogrid[:img_gray.shape[0], :img_gray.shape[1]]
    mask = (x - xc) ** 2 + (y - yc) ** 2 > r ** 2
    inside = np.ma.masked_where(mask, img_gray)
    average_color = inside.mean()
    return "Not Cataract" if average_color <= 60 else "Cataract"


def detect_cataract(filepath: str):
    """
    Detects cataract from an eye image.

    Parameters:
    - filepath (str): The path to the image file.

    Returns:
    - None
    """
    img_bgr, img_gray = read_image(filepath)
    img_bgr, img_gray, factor = resize_and_preprocess(img_gray, img_bgr)
    circles = detect_and_draw_circles(img_gray, img_bgr, factor)
    if circles is not None:
        message = evaluate_cataract(img_gray, circles)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_bgr, message, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        print(message)
    else:
        print("No circles detected.")
    cv2.imshow('Cataract Detection', img_bgr)
    cv2.waitKey(0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        detect_cataract(image_path)
    else:
        print("Please provide a path to an image.")

