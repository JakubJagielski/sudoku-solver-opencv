import cv2
import numpy as np
import stack_images as si
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print ("starting...")

import sudoku_engine as se

# make board sharper and more readable
def preProcessBoard(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold

def reorderCorners(corners):
    corners = corners.reshape((4, 2))
    fixedCorners = np.zeros((4, 1, 2), np.int32)

    add = corners.sum(1)
    #print (add)

    fixedCorners[0] = corners[np.argmin(add)]
    fixedCorners[3] = corners[np.argmax(add)]

    diff = np.diff(corners, axis=1)
    fixedCorners[1] = corners[np.argmin(diff)]
    fixedCorners[2] = corners[np.argmax(diff)]

    return fixedCorners

def biggestContour(contours):
    biggest_contour = np.array([])
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)
            if area > max_area and len(approx) == 4: # find largest squares
                max_area = area
                biggest_contour = approx

    return biggest_contour, max_area

# simply split grid and extract cells
def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for box in cols:
            boxes.append(box)
    return boxes

# recognise numbers (accepting based on threshold)
def getPrediction(boxes, model):
    result70 = []
    result80 = []
    result90 = []
    
    for image in boxes:
        ## prepare image
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # remove edges - only leave number
        img = np.asarray(img)
        img = img[4:img.shape[0]-4, 4:img.shape[1]-4]
        img = cv2.resize(img, (32, 32))
        img = img / 255
        img = img.reshape(1, 32, 32, 1)

        # predictions
        predictions = model.predict(img)
        class_index = np.argmax(predictions, axis=1)
        prob = np.amax(predictions)
        # print (f'{class_index}, {prob}')
        
        #result70.append(class_index[0])
        if prob > 0.7:
            result70.append(class_index[0])
        else:
            result70.append(0)
            
        if prob > 0.8:
            result80.append(class_index[0])
        else:
            result80.append(0)
            
        if prob > 0.9:
            result90.append(class_index[0])
        else:
            result90.append(0)
            
    return [result70, result80, result90]

def displayNumbers(img, numbers, color=(0, 255, 0)):
    secW = img.shape[1] // 9
    secH = img.shape[0] // 9
    for x in range (0, 9):
        for y in range (0, 9):
            if numbers[(y*9)+x] != 0:
                #cv2.putText(img, str(numbers[(y*9)+x]),
                #(x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_PLAIN,
                #            3, color, 2, cv2.LINE_AA)
                cv2.putText(img, str(numbers[(y * 9) + x]),
                            (int((x + 0.2) * secW), int((y + 0.85) * secH)), cv2.FONT_HERSHEY_PLAIN,
                            3, color, 2, cv2.LINE_AA)
    return img

def drawGrid(img, color=(0, 0, 255)):
    secW = img.shape[1] // 9
    secH = img.shape[0] // 9

    for i in range (0, 9): ## thin grid
        pt1 = (0, secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW*i, 0)
        pt4 = (secW*i, img.shape[0])
        thickness = 2
        if i % 3 == 0:
            thickness = 5
        cv2.line(img, pt1, pt2, color, thickness)
        cv2.line(img, pt3, pt4, color, thickness)
    return img

def highlightBoxes(img, mask, color=(0, 255, 0)):
    secW = img.shape[1] // 9
    secH = img.shape[0] // 9

    # blank image
    img_highlight = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    for i in range(0, 9):
        for j in range(0, 9):
            if mask[(i * 9) + j] == 0:
                ## draw green box
                cv2.rectangle(img_highlight, (j*secW, i*secH), ((j+1)*secW, (i+1)*secH),
                              color, cv2.FILLED)
    img_highlight = cv2.addWeighted(img_highlight, 0.3, img, 1, 1)
    for i in range(0, 9):
        for j in range(0, 9):
            if mask[(i * 9) + j] == 0:
                ## draw blue box
                cv2.rectangle(img_highlight, (j * secW, i * secH), ((j + 1) * secW, (i + 1) * secH),
                              (255, 0, 0), 3)
    return img_highlight


def main_process(data):    
    ## fetch blank images to work with
    img_detected_digits = data["img_detected_digits"]
    img_warped = data["img_warped"]

    sudoku_corners = reorderCorners(data["contour_corners"])
    cv2.drawContours(data["img_contours"], sudoku_corners, -1, (0, 0, 255), 10)

    # warp perspective
    pts_sudoku = np.float32(sudoku_corners)
    pts_target = np.float32([[0, 0], [data["width_img"], 0], [0, data["height_img"]],
                             [data["width_img"], data["height_img"]]])

    warp_matrix = cv2.getPerspectiveTransform(pts_sudoku, pts_target)
    img_warped = cv2.warpPerspective(data["img_original"], warp_matrix, (data["width_img"], data["height_img"]))

    img_solved_digits = img_warped.copy()

    boxes = splitBoxes(img_warped)
    
    # predict numbers using neural network
    numbers_sets = getPrediction(boxes, data["model"])
    #print (numbers_sets)
    numbers = numbers_sets [-1] # assume most accurate prediction is used (failed earlier iterations)
    pos_array = []
    board_solved = False
    sudoku_board = []
    
    # try progressively more accurate neural network predictions
    for numbers_iter in numbers_sets:
        numbers = np.asarray(numbers_iter)
        #print (numbers)

        # mask: leave only empty spaces as 1
        pos_array = np.where(numbers > 0, 0, 1)  # mask
        
        # build board from extracted information
        sudoku_board = []
        for i in range(0, 9):
            row = []
            for j in range(0, 9):
                num = numbers[(i * 9) + j]
                row.append(num)
            sudoku_board.append(row)
        
        # if no numbers are found, return board not identified
        if (all(num == 1 for num in pos_array)):
            #data["status"] = False
            continue
         
        # se.print_board(sudoku_board)
        if se.solve(sudoku_board):
            board_solved = True
            break
    
    data["status"] = board_solved
    
    # show digits recognised
    displayNumbers(img_detected_digits, numbers)
    img_detected_digits = drawGrid(img_detected_digits, color=(0, 255, 0))
    data["img_detected_digits"] = img_detected_digits
    
    ## highlight recognised digits
    img_warped = highlightBoxes(img_warped, pos_array)
    displayNumbers(img_warped, numbers, color=(255, 0, 0))
    data["img_warped"] = img_warped
    
    if board_solved == False: # if 70, 80 and 90% accuracy failed, return
        return data

    flat_list = []
    for sub_list in sudoku_board:
        for item in sub_list:
            flat_list.append(item)
    solved_nums = flat_list * pos_array

    ## draw solved numbers and grid
    img_solved_digits = displayNumbers(img_solved_digits, solved_nums, color=(0, 255//2, 0))
    img_solved_digits = drawGrid(img_solved_digits, color=(0, 255//2, 0))

    # overlay solution
    pts2 = np.float32(sudoku_corners)
    pts1 = np.float32([[0, 0], [data["width_img"], 0], [0, data["height_img"]],
                       [data["width_img"], data["height_img"]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    img_inv_warp_coloured = data["img_original"].copy()
    img_inv_warp_coloured = cv2.warpPerspective(img_solved_digits, matrix,
                                                (data["width_img"], data["height_img"]))

    inv_perspective = cv2.addWeighted(img_inv_warp_coloured, 1, data["img_original"], 0.3, 1)

    
    data["inv_perspective"] = inv_perspective
    data["img_solved_digits"] = img_solved_digits

    return data

def main(file_dir):
    #################
    # img dimensions
    width_img = 450
    height_img = 450

    # fetch image to analyze
    '''
    filename = ""
    try:
        filename = os.listdir("resources/input/")[2]
    except IndexError:
        print("No file found.")
        return
    '''
    
    img_raw = cv2.imread(file_dir)

    # preprocess image, to extract data from cells
    img_original = img_raw.copy()
    img_original = cv2.resize(img_original, (width_img, height_img))
    img_threshold = preProcessBoard(img_original)

    # load neural network - OCR.
    model = load_model("model_trained.h5")

    # find and draw contours
    contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = img_original.copy()
    cv2.drawContours(img_contours, contours, -1, (255//2, 0, 0), 3)

    # find grid from contours
    contour_corners, contour_area = biggestContour(contours)
    
    state = "None"
    
    # create analysis images to display at end
    img_blank = np.zeros((height_img, width_img, 3), np.uint8)
    
    # data to pass to main_process function
    data_input = {"width_img": width_img, "height_img": height_img, "model": model,
                  "contour_corners": contour_corners, "img_contours": img_contours,
                  "img_original": img_original, "img_detected_digits": img_blank.copy(),
                  "img_warped": img_blank.copy(), "inv_perspective": img_blank.copy(),
                  "img_solved_digits": img_blank.copy()
                  }
                  
    output = data_input
    #output = main_process(data_input) # output
    
    if contour_corners.size != 0:  # if grid found....
        output = main_process(data_input) # output
        if output["status"]:
            # display results
            
            #cv2.imshow("sudoku", img_analysis)              
            #cv2.imshow("final", result)
            
            
            #cv2.waitKey(0)
            state = "solved"
        else:
            state = "not solved"
            #print("Board could not be solved.")
    else:
        state = "not found"
        #print("Sudoku Grid Not Found!")
        
    # display results (write analysis image to file)
    img_analysis = si.stackImages(0.6, ([img_original, img_threshold, img_contours],
                                  [output["img_warped"], output["img_detected_digits"],
                                   output["img_solved_digits"]]))
                                       
    result = cv2.resize(output["inv_perspective"], (img_raw.shape[1], img_raw.shape[0]),
                        interpolation=cv2.INTER_AREA)
                        
    cv2.imwrite(f'static/results/analysis.jpg', img_analysis)
    cv2.imwrite(f'static/results/result.jpg', result)
    
    return state
    
if __name__ == '__main__':
    main("board_2.jpeg")

