def print_board(board):
    for row in board:
        print (row)

def isValid(board, row, column, number):
    # check columns for specific row
    for col_i in range(0, 9):
        # if col_i != column:
        if board[row][col_i] == number:
            return False

    # check column
    for row_i in range(0, 9):
        # if row_i != row:
        if board[row_i][column] == number:
            return False

    ## check sub grid
    g_row = row - row % 3
    g_col = column - column % 3

    #print (f'row:{g_row} col:{g_col}')

    for i in range(g_row, g_row+3):
        for k in range(g_col, g_col+3):
            if board[i][k] == number:
                return False
    return True

def solve(board):
    # loop through the board:
    for row_i in range(0, 9):
        for col_i in range(0, 9):
            if board[row_i][col_i] == 0:
                for num in range (1, 10):
                    if (isValid(board, row_i, col_i, num)):
                        board[row_i][col_i] = num
                        if solve(board):
                            return True
                        else:
                            #return False
                            board[row_i][col_i] = 0
                return False
    return True

if __name__ == '__main__':
    board = [
        [0, 4, 0, 7, 0, 0, 1, 3, 0],
        [0, 0, 2, 0, 0, 0, 6, 0, 0],
        [0, 0, 0, 4, 2, 0, 0, 0, 0],
        [6, 0, 0, 0, 0, 2, 0, 0, 3],
        [2, 3, 1, 0, 7, 0, 0, 8, 0],
        [4, 0, 0, 3, 1, 0, 0, 0, 0],
        [0, 7, 0, 0, 0, 8, 0, 0, 0],
        [0, 0, 6, 0, 3, 0, 0, 0, 4],
        [8, 9, 0, 0, 5, 0, 0, 0, 6]
    ]
    if (solve(board)):
        print ("solved")
        print_board(board)
    else:
        print ("board not solved")

