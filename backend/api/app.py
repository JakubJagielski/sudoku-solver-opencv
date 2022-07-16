import fastapi
import api.service as service


app = fastapi.FastAPI()


@app.get("/healthcheck")
def health_check():
    return "alive"


@app.get("/solve")
def solve_sudoku_from_string(board: str = fastapi.Query(regex="^[0-9]{5}$")):
    return service.solve_sudoku_from_string_api(board)
