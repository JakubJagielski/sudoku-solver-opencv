import backend.api.service as service
import backend.api.sudoku_solver_visual as sudoku_solver_visual
import cv2
import fastapi
import fastapi.responses
import numpy as np

app = fastapi.FastAPI()


@app.get("/healthcheck")
def health_check():
    return "alive"


@app.get("/solve", responses={400: {"description": "Unsolvable board"}})
async def solve_sudoku_from_string(
    board: str = fastapi.Query(
        default="006003470700090500080001063008002001001807200600300800150700080007080009024100700",
        regex="^[0-9]{81}$",
    )
):
    return service.solve_sudoku_from_string(board)


@app.post("/solve-image")
async def solve_sudoku_from_image(
    file: fastapi.UploadFile,
    sudoku_extractor: sudoku_solver_visual.SudokuExtractor = fastapi.Depends(
        sudoku_solver_visual.extract_sudoku_string_from_grid
    ),
):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image: np.ndarray = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    solution_image = sudoku_solver_visual.solve_sudoku_from_image(
        image, sudoku_extractor
    )

    solution_image_bytes = cv2.imencode(".jpg", solution_image)[1].tostring()
    return fastapi.Response(content=solution_image_bytes, media_type="image/jpeg")
