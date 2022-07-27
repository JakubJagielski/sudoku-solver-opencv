# sudoku-solver-opencv

How to run ?

>> `docker build . sudoku_api`
>> `docker run -p 80:80 sudoku_api`


## Description
An API which solves sudoku puzzles from images.

endpoint `POST /solve-image`

- accepts image file
- processes it to make it more readable
- identifies the sudoku grid
- warps the grid to perfect square
- splits grid into cells
- performts image recognition on each cell and identifies the digits
- combines digits into a string representation of a sudoku puzzle
- solves the sudoku
- overwrites the image, filling in the solved digits
- unwarps the image and overlays it on the original image
- returns the solved image

endpoint `GET /solve?board=...`
- accepts a string representation of sudoku
- solves it
- returns solution string