import typing as t

import backend.api.schemas as schemas
import backend.api.service as service
import cv2
import fastapi
import numpy as np

WIDTH_IMG = 450
HEIGHT_IMG = 450


def find_corners_of_biggest_contour(contours) -> np.array | None:
    biggest_contour_corners = None
    biggest_area_found = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:
            continue

        perimeter = cv2.arcLength(contour, True)
        contour_corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if (
            area > biggest_area_found and len(contour_corners) == 4
        ):  # find largest squares
            biggest_area_found = area
            biggest_contour_corners = contour_corners

    return biggest_contour_corners


def reorder_grid_corners(shuffled_corners: np.array) -> np.array:
    shuffled_corners = shuffled_corners.reshape((4, 2))
    reordered_corners = np.zeros((4, 1, 2), np.int32)

    add = shuffled_corners.sum(1)
    reordered_corners[0] = shuffled_corners[np.argmin(add)]
    reordered_corners[3] = shuffled_corners[np.argmax(add)]

    diff = np.diff(shuffled_corners, axis=1)
    reordered_corners[1] = shuffled_corners[np.argmin(diff)]
    reordered_corners[2] = shuffled_corners[np.argmax(diff)]

    return reordered_corners


def find_grid_corners_from_contours(contours) -> np.array | None:
    shuffled_grid_corners = find_corners_of_biggest_contour(contours)
    if shuffled_grid_corners is None:
        return None
    return reorder_grid_corners(shuffled_grid_corners)


def solve_sudoku_from_image(sudoku_image_raw: np.ndarray) -> np.ndarray:
    sudoku_image_original = sudoku_image_raw.copy()

    sudoku_image_sharpened = pre_process_board(sudoku_image_original)
    sudoku_grid_image_warped, grid_corners = extract_grid_and_corners_from_sudoku_image(
        sudoku_image_sharpened, sudoku_image_raw
    )

    sudoku_puzzle = extract_sudoku_string_from_grid_image(
        sudoku_grid_image_warped
    )  # neural network

    sudoku_solution = service.solve_sudoku_from_string(sudoku_puzzle)

    # convert solution to image
    # unwarp image to fit original image
    # return solved, unwarped image

    result_image = build_response_image(sudoku_solution, sudoku_image_original)


def split_boxes(sudoku_grid_image: np.ndarray) -> list[np.ndarray]:
    return [box for row in np.vsplit(sudoku_grid_image) for box in np.hsplit(row, 9)]


def prepare_image_for_classification(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove edges - only leave number
    img = np.asarray(img)
    img = img[4 : img.shape[0] - 4, 4 : img.shape[1] - 4]
    img = cv2.resize(img, (32, 32))
    img = img / 255
    img = img.reshape(1, 32, 32, 1)
    return img


def classify_digit(image: np.ndarray, model) -> int:
    predictions = model.predict(image)
    class_index = np.argmax(predictions, axis=1)
    return class_index[0]


def extract_sudoku_string_from_grid_image(sudoku_grid_image: np.ndarray) -> str:
    # break down image into cells
    cells = split_boxes(sudoku_grid_image)
    cells = map(prepare_image_for_classification, cells)
    sudoku_puzzle = map(classify_digit, cells)
    return "".join([str(digit) for digit in sudoku_puzzle])


def extract_grid_and_corners_from_sudoku_image(
    sudoku_image_sharp: np.ndarray,
    sudoku_image_original: np.ndarray,
) -> t.Tuple[np.ndarray, np.array]:
    contours, _ = cv2.findContours(
        sudoku_image_sharp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    grid_corners = find_grid_corners_from_contours(contours)

    if grid_corners is None:
        return None, None

    pts_sudoku = np.float32(grid_corners)
    pts_target = np.float32(
        [[0, 0], [WIDTH_IMG, 0], [0, HEIGHT_IMG], [WIDTH_IMG, HEIGHT_IMG]]
    )

    warp_matrix = cv2.getPerspectiveTransform(pts_sudoku, pts_target)
    img_warped = cv2.warpPerspective(
        sudoku_image_original, warp_matrix, (WIDTH_IMG, HEIGHT_IMG)
    )

    return img_warped, grid_corners


def pre_process_board(sudoku_grid_image: np.ndarray) -> np.ndarray:
    img_gray = cv2.cvtColor(sudoku_grid_image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)
    return img_threshold


def build_response_image(sudoku_solution: str, image: np.ndarray) -> np.ndarray:
    pass
