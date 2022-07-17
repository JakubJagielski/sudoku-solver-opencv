import backend.api.service as service
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
async def solve_sudoku_from_image(file: fastapi.UploadFile):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img: np.ndarray = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img = process(img)
    img_str = cv2.imencode(".jpg", img)[1].tostring()
    return fastapi.Response(content=img_str, media_type="image/jpeg")
