import backend.tests.conftest as conftest
import starlette.testclient


def test_solve_board_endpoint(client: starlette.testclient.TestClient) -> None:
    response = client.get("/solve", params={"board": conftest.STRING_BOARD})
    assert response.status_code == 200
    assert response.json() == conftest.STRING_SOLUTION


def test_solve_board_endpoint_fails_on_unsolvable_board(
    client: starlette.testclient.TestClient,
) -> None:
    response = client.get("/solve", params={"board": conftest.UNSOLVABLE_STRING_BOARD})
    assert response.status_code == 400


def test_solve_image_endpoint(
    client_with_solvable_identified_sudoku: starlette.testclient.TestClient,
) -> None:
    response = client_with_solvable_identified_sudoku.post(
        "/solve-image",
        files={"file": ("filename", open(conftest.SUDOKU_FILENAME, "rb"), "image/jpg")},
    )
    assert response.status_code == 200


def test_solve_image_endpoint_no_puzzle_identified(
    client_with_empty_identified_sudoku: starlette.testclient.TestClient,
) -> None:
    response = client_with_empty_identified_sudoku.post(
        "/solve-image",
        files={"file": ("filename", open(conftest.SUDOKU_FILENAME, "rb"), "image/jpg")},
    )
    assert response.status_code == 422
