import fastapi
import api.schemas as schemas


def solve_sudoku(sudoku_board: schemas.Board) -> bool:
    for row_index in range(0, 9):
        for column_index in range(0, 9):
            if sudoku_board.get(row_index, column_index) == 0:
                for trial_number in range(1, 10):
                    if sudoku_board.is_valid_entry(
                        row_index, column_index, trial_number
                    ):
                        sudoku_board.set(row_index, column_index, trial_number)
                        if solve_sudoku(sudoku_board):
                            return True
                        else:
                            sudoku_board.set(row_index, column_index, 0)
                return False
    return True


def solve_sudoku_from_string(board: str) -> str | None:
    sudoku_board = schemas.Board(board_content=board)
    success = solve_sudoku(sudoku_board)
    if success:
        return sudoku_board.as_string()
    return None


def solve_sudoku_from_string_api(board: str) -> str:
    solution = solve_sudoku_from_string(board)
    if solution is None:
        raise fastapi.exceptions.HTTPException(
            status_code=400, detail="Sudoku could not be solved"
        )
    return solution
