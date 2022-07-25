import itertools
import os
import typing as t

import backend.api.schemas as schemas
import backend.api.service as service
import cv2
import fastapi
import numpy as np
import tensorflow.keras.models

WIDTH_IMG = 450
HEIGHT_IMG = 450

MODEL = tensorflow.keras.models.load_model("backend/api/model_trained.h5")


def find_corners_of_biggest_contour(contours) -> np.ndarray | None:
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


def reorder_grid_corners(shuffled_corners: np.array) -> np.ndarray:
    shuffled_corners = shuffled_corners.reshape((4, 2))
    reordered_corners = np.zeros((4, 1, 2), np.int32)

    add = shuffled_corners.sum(1)
    reordered_corners[0] = shuffled_corners[np.argmin(add)]
    reordered_corners[3] = shuffled_corners[np.argmax(add)]

    diff = np.diff(shuffled_corners, axis=1)
    reordered_corners[1] = shuffled_corners[np.argmin(diff)]
    reordered_corners[2] = shuffled_corners[np.argmax(diff)]

    return reordered_corners


def find_grid_corners_from_contours(contours) -> np.ndarray | None:
    shuffled_grid_corners = find_corners_of_biggest_contour(contours)
    if shuffled_grid_corners is None:
        return None
    return reorder_grid_corners(shuffled_grid_corners)


def write_numbers_to_img(
    img: np.ndarray, numbers: str | list[int], colour=(0, 255 // 2, 0)
) -> np.ndarray:
    img_copy = img.copy()
    section_width = img_copy.shape[1] // 9
    section_height = img_copy.shape[0] // 9
    for column, row in itertools.product(range(9), range(9)):
        if int(numbers[(row * 9) + column]) == 0:
            continue

        cv2.putText(
            img_copy,
            str(numbers[(row * 9) + column]),
            (int((column + 0.2) * section_width), int((row + 0.85) * section_height)),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            colour,
            2,
            cv2.LINE_AA,
        )
    return img_copy


def draw_grid(img: np.ndarray, colour=(0, 255 // 2, 0)) -> np.ndarray:
    img_copy = img.copy()
    section_width = img_copy.shape[1] // 9
    section_height = img_copy.shape[0] // 9

    for i in range(0, 9):
        pt1 = (0, section_height * i)
        pt2 = (img_copy.shape[1], section_height * i)
        pt3 = (section_width * i, 0)
        pt4 = (section_width * i, img_copy.shape[0])
        thickness = 2
        if i % 3 == 0:
            thickness = 5
        cv2.line(img_copy, pt1, pt2, colour, thickness)
        cv2.line(img_copy, pt3, pt4, colour, thickness)
    return img_copy


def construct_sudoku_representation(
    img_to_overwrite: np.ndarray,
    digits_to_display: list[int] | str,
    colour=(0, 255 // 2, 0),
) -> np.ndarray:
    img_with_grid = draw_grid(img_to_overwrite, colour)
    return write_numbers_to_img(img_with_grid, digits_to_display, colour)


def solve_sudoku_from_image(sudoku_image_raw: np.ndarray) -> np.ndarray:
    sudoku_image_original = sudoku_image_raw.copy()
    sudoku_image_sharpened = pre_process_board(sudoku_image_original)
    sudoku_grid_image_warped, grid_corners = extract_grid_and_corners_from_sudoku_image(
        sudoku_image_sharpened, sudoku_image_raw
    )

    sudoku_puzzle = extract_sudoku_string_from_grid_image(sudoku_grid_image_warped)

    if sudoku_puzzle == schemas.EMPTY_PUZZLE:
        raise fastapi.exceptions.HTTPException(422, "Sudoku not identified from image")

    sudoku_solution, sudoku_success = service.solve_board(
        schemas.Board.from_string(sudoku_puzzle)
    )
    if sudoku_success is False:
        return construct_sudoku_representation(
            np.zeros((HEIGHT_IMG, WIDTH_IMG, 3), np.uint8),
            sudoku_puzzle,
            (255, 0, 0),
        )

    sudoku_solution_as_string = sudoku_solution.as_string()
    digits_filled_in_by_algorithm = [
        int(sudoku_solution_as_string[i]) if int(sudoku_puzzle[i]) == 0 else 0
        for i in range(0, 81)
    ]

    solution_image = construct_sudoku_representation(
        sudoku_grid_image_warped.copy(), digits_filled_in_by_algorithm
    )

    solution_image_to_overlay = unwarp_image(
        solution_image,
        grid_corners,
        sudoku_image_original.shape[1],
        sudoku_image_original.shape[0],
    )

    final_image = cv2.addWeighted(
        solution_image_to_overlay, 1, sudoku_image_original, 0.3, 1
    )
    return cv2.resize(final_image, (WIDTH_IMG, HEIGHT_IMG))


def unwarp_image(
    solution_image: np.ndarray,
    grid_corners,
    width_original: float,
    height_original: float,
) -> np.ndarray:
    solution_image_copy = solution_image.copy()
    unwarp_matrix = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [WIDTH_IMG, 0], [0, HEIGHT_IMG], [WIDTH_IMG, HEIGHT_IMG]]),
        np.float32(grid_corners),
    )
    return cv2.warpPerspective(
        solution_image_copy, unwarp_matrix, (width_original, height_original)
    )


def split_boxes(sudoku_grid_image: np.ndarray) -> list[np.ndarray]:
    return [box for row in np.vsplit(sudoku_grid_image, 9) for box in np.hsplit(row, 9)]


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


def classify_digit(image: np.ndarray) -> int:
    predictions = MODEL.predict(image)
    class_index = np.argmax(predictions, axis=1)
    probability = np.amax(predictions)
    if probability > 0.7:
        return class_index[0]
    return 0


def extract_sudoku_string_from_grid_image(sudoku_grid_image: np.ndarray) -> str:
    cells = split_boxes(sudoku_grid_image)
    cells = map(prepare_image_for_classification, cells)
    sudoku_puzzle = map(classify_digit, cells)
    return "".join([str(digit) for digit in sudoku_puzzle])


def extract_grid_and_corners_from_sudoku_image(
    sudoku_image_sharp: np.ndarray,
    sudoku_image_original: np.ndarray,
) -> t.Tuple[np.ndarray | None, np.ndarray | None]:
    contours, _ = cv2.findContours(
        sudoku_image_sharp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    grid_corners = find_grid_corners_from_contours(contours)

    if grid_corners is None:
        raise fastapi.exceptions.HTTPException(
            422, "Cannot find grid corners from image."
        )

    warp_matrix = cv2.getPerspectiveTransform(
        np.float32(grid_corners),
        np.float32([[0, 0], [WIDTH_IMG, 0], [0, HEIGHT_IMG], [WIDTH_IMG, HEIGHT_IMG]]),
    )
    img_warped = cv2.warpPerspective(
        sudoku_image_original, warp_matrix, (WIDTH_IMG, HEIGHT_IMG)
    )

    return img_warped, grid_corners


def pre_process_board(sudoku_grid_image: np.ndarray) -> np.ndarray:
    img_gray = cv2.cvtColor(sudoku_grid_image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 11, 2)
    return img_threshold
