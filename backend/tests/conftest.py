import backend.api.app
import backend.api.schemas as schemas
import backend.api.sudoku_solver_visual as sudoku_solver_visual
import pytest
import starlette.testclient

SUDOKU_FILENAME = "backend/api/sample_images/sample_sudoku.jpg"

STRING_BOARD = (
    "006003470700090500080001063008002001001807200600300800150700080007080009024100700"
)
STRING_SOLUTION = (
    "916253478743698512285471963578942631431867295692315847159726384367584129824139756"
)
UNSOLVABLE_STRING_BOARD = (
    "126003470700090500080001063008002001001807200600300800150700080007080009024100700"
)

BOARD = schemas.Board(
    [
        [0, 0, 6, 0, 0, 3, 4, 7, 0],
        [7, 0, 0, 0, 9, 0, 5, 0, 0],
        [0, 8, 0, 0, 0, 1, 0, 6, 3],
        [0, 0, 8, 0, 0, 2, 0, 0, 1],
        [0, 0, 1, 8, 0, 7, 2, 0, 0],
        [6, 0, 0, 3, 0, 0, 8, 0, 0],
        [1, 5, 0, 7, 0, 0, 0, 8, 0],
        [0, 0, 7, 0, 8, 0, 0, 0, 9],
        [0, 2, 4, 1, 0, 0, 7, 0, 0],
    ]
)


SOLUTION = schemas.Board(
    [
        [9, 1, 6, 2, 5, 3, 4, 7, 8],
        [7, 4, 3, 6, 9, 8, 5, 1, 2],
        [2, 8, 5, 4, 7, 1, 9, 6, 3],
        [5, 7, 8, 9, 4, 2, 6, 3, 1],
        [4, 3, 1, 8, 6, 7, 2, 9, 5],
        [6, 9, 2, 3, 1, 5, 8, 4, 7],
        [1, 5, 9, 7, 2, 6, 3, 8, 4],
        [3, 6, 7, 5, 8, 4, 1, 2, 9],
        [8, 2, 4, 1, 3, 9, 7, 5, 6],
    ]
)

UNSOLVABLE_BOARD = schemas.Board(
    [
        [1, 2, 6, 0, 0, 3, 4, 7, 0],
        [7, 0, 0, 0, 9, 0, 5, 0, 0],
        [0, 8, 0, 0, 0, 1, 0, 6, 3],
        [0, 0, 8, 0, 0, 2, 0, 0, 1],
        [0, 3, 1, 8, 0, 7, 2, 0, 0],
        [6, 0, 0, 3, 0, 0, 8, 0, 0],
        [1, 5, 0, 7, 0, 0, 0, 8, 0],
        [0, 0, 7, 0, 8, 0, 0, 0, 9],
        [0, 2, 4, 1, 0, 0, 7, 0, 0],
    ]
)


@pytest.fixture
def client() -> starlette.testclient.TestClient:
    return starlette.testclient.TestClient(backend.api.app.app)


@pytest.fixture
def client_with_solvable_identified_sudoku() -> starlette.testclient.TestClient:
    app = backend.api.app.app
    app.dependency_overrides[
        sudoku_solver_visual.extract_sudoku_string_from_grid
    ] = lambda: lambda x: STRING_BOARD
    return starlette.testclient.TestClient(app)


@pytest.fixture
def client_with_empty_identified_sudoku() -> starlette.testclient.TestClient:
    app = backend.api.app.app
    app.dependency_overrides[
        sudoku_solver_visual.extract_sudoku_string_from_grid
    ] = lambda: lambda x: schemas.EMPTY_PUZZLE
    return starlette.testclient.TestClient(app)
