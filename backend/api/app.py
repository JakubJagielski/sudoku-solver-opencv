import fastapi


app = fastapi.FastAPI()


@app.get("/healthcheck")
def health_check():
    return "alive"


@app.get("/solve")
def solve_sudoku(board: str = fastapi.Query(regex="^[0-9]{5}$")):
    return board
