import backend.api.schemas as schemas
import backend.tests.conftest as conftest


def test_board_from_string() -> None:
    assert schemas.Board.from_string(conftest.STRING_BOARD) == conftest.BOARD


def test_board_as_string() -> None:
    assert conftest.BOARD.as_string() == conftest.STRING_BOARD
