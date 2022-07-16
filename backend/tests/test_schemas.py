import api.schemas as schemas

STRING_BOARD = (
    "006003478700090500080001063008002001001807200600300400150700080007080009024100700"
)
BOARD = [
    [0, 0, 6, 0, 0, 3, 4, 7, 8],
    [7, 0, 0, 0, 9, 0, 5, 0, 0],
    [0, 8, 0, 0, 0, 1, 0, 6, 3],
    [0, 0, 8, 0, 0, 2, 0, 0, 1],
    [0, 0, 1, 8, 0, 7, 2, 0, 0],
    [6, 0, 0, 3, 0, 0, 4, 0, 0],
    [1, 5, 0, 7, 0, 0, 0, 8, 0],
    [0, 0, 7, 0, 8, 0, 0, 0, 9],
    [0, 2, 4, 1, 0, 0, 7, 0, 0],
]

SOLUTION = (
    "916253478743698512285471963578942631431867295692315847159726384367584129824139756"
)


def test_board_init() -> None:
    assert schemas.Board(STRING_BOARD).get_board_raw() == BOARD


def test_board_as_string() -> None:
    assert schemas.Board(STRING_BOARD).as_string() == STRING_BOARD


def test_board_is_valid_entry() -> None:
    assert False
