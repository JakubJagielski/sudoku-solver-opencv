import fastapi
import backend.api.service as service


app = fastapi.FastAPI()


@app.get("/healthcheck")
def health_check():
    return "alive"


@app.get("/solve", responses={400: {"description": "Unsolvable board"}})
def solve_sudoku_from_string(board: str = fastapi.Query(regex="^[0-9]{81}$")):
    return service.solve_sudoku_from_string(board)
