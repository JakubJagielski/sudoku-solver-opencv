# sudoku-solver-opencv

See a live demo at: https://jakub-sudoku.herokuapp.com

A web application written in Python and deployed using Flask. Upload a photo of a sudoku puzzle, and the application solves it. It manipulates images using OpenCV, a computer vision package, and uses a self trained neural network for optical character recognition, to understand the input. After solving the puzzle, the app overwrites the original image, filling in the missing digits. 

Steps:

- itentify the board
- warp board to perpendicular view
- split warped board into cells
- do optical character recognition on each cell, using a self trained neural network
- solve the puzzle using a self made sudoku engine
- overwrite the original image with the results
