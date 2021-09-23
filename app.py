import os, sys
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
import sudoku_solver_visual



UPLOAD_FOLDER  = "static/uploads"
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

app = Flask (__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


import logging

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
     
@app.route("/sample_sudoku", methods=['GET', 'POST'])
def sample():
    if request.method == 'POST':
        sample_image_path = "static/sample_sudokus/sample_1.jpg"
        
        if request.form.get('action1') == 'Sudoku_1':
           sample_image_path = "static/sample_sudokus/sample_1.jpg"
        elif  request.form.get('action2') == 'Sudoku_2':
            sample_image_path = "static/sample_sudokus/sample_2.png"
        elif  request.form.get('action3') == 'Sudoku_3':
            sample_image_path = "static/sample_sudokus/sample_3.jpg"
        elif request.form.get('action_home') == 'HOME':
            return redirect("/")
        else:
            return "UNKNOWN"
        
        ## clear upload and result directory
        for f in os.listdir("static/uploads"):
            if f != "empty.txt":  ## leave empty text file for git reasons
                os.remove(os.path.join("static/uploads", f))
        for f in os.listdir("static/results"):
            os.remove(os.path.join("static/results", f))
        
        ## call main.py (filename)
        state = sudoku_solver_visual.main(sample_image_path)
        
        if state == "solved":
            result_filename = "static/results/result.jpg"
            analysis_filename = "static/results/analysis.jpg"
            return render_template("result.html", data = [result_filename, analysis_filename])
        else:
            return "error"
            
    data = {"sample_1": "sample_sudokus/sample_1.jpg", "sample_2": "sample_sudokus/sample_2.png",
        "sample_3": "sample_sudokus/sample_3.jpg"}
    return render_template('index.html', data = data)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the POST request has a file
        if 'file' not in request.files:
            return redirect(url_for('index'))
        # else
        file = request.files['file']
        # if user doesn't select a file, browser submits an empty file without a name
        if file.filename == '':
            return redirect(url_for('index'))
        if file and allowed_file(file.filename):
            ## clear upload and result directory
            for f in os.listdir("static/uploads"):
                if f != "empty.txt":  ## leave empty text file for git reasons
                    os.remove(os.path.join("static/uploads", f))
            for f in os.listdir("static/results"):
                os.remove(os.path.join("static/results", f))
        
            filename = secure_filename(file.filename)
            upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_filepath)
            
            
            ## call main.py (filename)
            state = sudoku_solver_visual.main(f'{UPLOAD_FOLDER}/{filename}')
            
            if state == "solved":
                result_filename = "static/results/result.jpg"
                analysis_filename = "static/results/analysis.jpg"
                return render_template("result.html", data = [result_filename, analysis_filename])
            else:
                message = "ERROR"
                if state == "not solved":
                    message = "Puzzle does not have a solution. Try a different puzzle."
                elif state == "not found":
                    message = "Board could not be identified. Try again with a clearer image."
                
                return render_template("failed.html", data = [message, upload_filepath])
                ## image couldn't be resolved. Try a new or clearer image.
    data = {"sample_1": "sample_sudokus/sample_1.jpg", "sample_2": "sample_sudokus/sample_2.png",
        "sample_3": "sample_sudokus/sample_3.jpg"}
    return render_template('index.html', data = data)
    
if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
    #app.run(debug=True)