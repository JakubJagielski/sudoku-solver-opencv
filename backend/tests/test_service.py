import backend.tests.conftest as conftest
import backend.api.service as service


def test_solve_board() -> None:
    new_board, success = service.solve_board(conftest.BOARD)
    assert success
    assert new_board == conftest.SOLUTION


def test_solve_board_fails_on_unsolvable_board() -> None:
    _, success = service.solve_board(conftest.UNSOLVABLE_BOARD)
    assert success is False
