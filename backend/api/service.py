import typing as t

import backend.api.schemas as schemas
import cv2
import fastapi
import numpy as np


def solve_board(sudoku_board: schemas.Board) -> t.Tuple[schemas.Board, bool]:
    sudoku_board_copy = sudoku_board.copy()

    def _solve(sudoku_board_copy: schemas.Board) -> bool:
        for row_index in range(0, 9):
            for column_index in range(0, 9):
                if sudoku_board_copy.get(row_index, column_index) != 0:
                    continue

                for trial_number in range(1, 10):
                    if not sudoku_board_copy.is_valid_entry(
                        row_index, column_index, trial_number
                    ):
                        continue

                    sudoku_board_copy.set(row_index, column_index, trial_number)
                    if _solve(sudoku_board_copy):
                        return True
                    sudoku_board_copy.set(row_index, column_index, 0)
                return False
        return True

    success = _solve(sudoku_board_copy)
    return sudoku_board_copy, success


def solve_sudoku_from_string(board: str) -> str:
    board_solution, success = solve_board(sudoku_board=schemas.Board.from_string(board))
    if not success:
        raise fastapi.exceptions.HTTPException(
            status_code=400, detail=f"Sudoku could not be solved: {board}"
        )
    return board_solution.as_string()
