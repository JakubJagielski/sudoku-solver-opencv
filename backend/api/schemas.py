from __future__ import annotations

import copy
import dataclasses


@dataclasses.dataclass
class Board:
    data: list[list[int]]

    def __init__(self, board_content: list[list[int]]):
        self.data = copy.deepcopy(board_content)

    @classmethod
    def from_string(
        cls,
        board_content: str = "000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    ):
        assert len(board_content) == 81
        content_as_array = []
        for row in range(0, 9):
            new_row = [int(number) for number in board_content[row * 9 : row * 9 + 9]]
            content_as_array.append(new_row)

        return cls(content_as_array)

    def copy(self) -> Board:
        return Board(board_content=copy.deepcopy(self.data))

    def get(self, row: int, col: int) -> int:
        return self.data[row][col]

    def set(self, row: int, col: int, value: int) -> None:
        self.data[row][col] = value

    def as_string(self) -> str:
        return "".join(str(column) for row in self.data for column in row)

    def is_valid_entry(self, row: int, column: int, value: int):
        for column_index in range(0, 9):
            if self.get(row=row, col=column_index) == value:
                return False

        for row_index in range(0, 9):
            if self.get(row=row_index, col=column) == value:
                return False

        ## check sub grid
        grid_row = row - row % 3
        grid_col = column - column % 3

        for row_index in range(grid_row, grid_row + 3):
            for column_index in range(grid_col, grid_col + 3):
                if self.get(row=row_index, col=column_index) == value:
                    return False
        return True

EMPTY_PUZZLE = "000000000000000000000000000000000000000000000000000000000000000000000000000000000"