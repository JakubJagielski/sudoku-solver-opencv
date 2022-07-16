import dataclasses


@dataclasses.dataclass
class Board:
    data: list[list[int]]

    def __init__(self, board_content: str):
        assert len(board_content) == 81
        self.data = []
        for row in range(0, 9):
            new_row = [int(number) for number in board_content[row * 9 : row * 9 + 9]]
            self.data.append(new_row)

    def get_board_raw(self) -> list[list[int]]:
        return self.data

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
