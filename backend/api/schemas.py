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


def solve_board(legal_board: str) -> str | None:
    return None
