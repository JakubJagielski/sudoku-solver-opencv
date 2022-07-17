import typing as t

import backend.api.schemas as schemas
import backend.api.service as service
import cv2
import fastapi
import numpy as np

WIDTH_IMG = 450
HEIGHT_IMG = 450


def biggest_contour(contours) -> t.Tuple[np.array, float]:
    biggest_contour = np.array([])
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if area > max_area and len(approx) == 4:  # find largest squares
            max_area = area
            biggest_contour = approx

    return biggest_contour, max_area


def solve_sudoku_from_image(sudoku_image_raw: np.ndarray) -> np.ndarray:
    sudoku_image = sudoku_image_raw.copy()

    sudoku_image = pre_process_board(sudoku_image)

    
    sudoku_puzzle = extract_sudoku_from_image(sudoku_image)

    sudoku_solution = service.solve_sudoku_from_string(sudoku_puzzle)
    result_image = build_response_image(sudoku_solution, sudoku_image)


def extract_sudoku_from_image(sudoku_image: np.ndarray) -> str:
    # find and draw contours
    contours, hierarchy = cv2.findContours(
        sudoku_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    img_contours = sudoku_image.copy()
    cv2.drawContours(img_contours, contours, -1, (255 // 2, 0, 0), 3)

    # find grid from contours
    contour_corners, contour_area = biggest_contour(contours)

    state = "None"

    # create analysis images to display at end
    img_blank = np.zeros((height_img, width_img, 3), np.uint8)

    # data to pass to main_process function
    data_input = {
        "width_img": width_img,
        "height_img": height_img,
        "model": model,
        "contour_corners": contour_corners,
        "img_contours": img_contours,
        "img_original": img_original,
        "img_detected_digits": img_blank.copy(),
        "img_warped": img_blank.copy(),
        "inv_perspective": img_blank.copy(),
        "img_solved_digits": img_blank.copy(),
    }

    output = data_input
    # output = main_process(data_input) # output

    if contour_corners.size != 0:  # if grid found....
        output = main_process(data_input)  # output
        if output["status"]:
            state = "solved"
        else:
            state = "not solved"
    else:
        state = "not found"


def pre_process_board(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (WIDTH_IMG, HEIGHT_IMG))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)
    return img_threshold


def build_response_image(sudoku_solution: str, image: np.ndarray) -> np.ndarray:
    pass
