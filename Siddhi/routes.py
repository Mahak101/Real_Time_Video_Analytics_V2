import os,re
from app import app
from flask import render_template,url_for,redirect,request,flash
import urllib.request
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

UPLOAD_FOLDER = 'static/uploads/'


 
app.secret_key = "siddhi-gupta"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

 

     
 
@app.route("/")
def view_home_page():
    return render_template("home.html", title="Home page")
  
@app.route("/Library")
def uploader():
        path = 'static/uploads/'
        uploads = sorted(os.listdir(path), key=lambda x: os.path.getctime(path+x))        # Sorting as per image upload date and time
        print(uploads)
        #uploads = os.listdir('static/uploads')
        uploads = ['uploads/' + file for file in uploads]
        uploads.reverse()
        return render_template("library.html",uploads=uploads)            # Pass filenames to front end for display in 'uploads' variable

app.config['UPLOAD_PATH'] =  UPLOAD_FOLDER            # Storage path    
@app.route("/",methods=['GET','POST'])
def upload_file():                                       # This method is used to upload files 
        if request.method == 'POST':
                f = request.files['file']
                #f.save(secure_filename(f.filename))
                filename = secure_filename(f.filename)
                f.save(os.path.join(app.config['UPLOAD_PATH'], filename))
                return redirect(url_for('view_library_page'))

@app.route("/Stream")
def view_stream_page():
    return render_template("stream.html", title="Stream page")
    

 

@app.route("/Visualize")
def view_visualize_page():
    return render_template("visualize.html", title="Visualize page")

@app.route("/Analyze")
def view_analyze_page():
    return render_template("analyze.html", title="Analyze page")

@app.route("/Models")
def view_models_page():
    return render_template("models.html", title="Models page")

@app.route("/Library")
def view_library_page():
    return render_template("library.html", title="Library page")

@app.route("/Devices")
def view_devices_page():
    return render_template("devices.html", title="Devices page")