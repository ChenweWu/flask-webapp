# from flask import Flask, request, render_template, send_from_directory
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import io
# import base64
# from flask import Flask, render_template, request, redirect, url_for
# import os
# # from predict import process_and_predict, plot_csv_data
# from werkzeug.utils import secure_filename
# from predict import *
# import time
# app = Flask(__name__)

# app.config['UPLOAD_FOLDER'] = 'uploads/'
# app.config['ALLOWED_EXTENSIONS'] = {'csv'}
# from flask import Flask, request, render_template, send_from_directory, session
# from flask_session import Session  # You might need to install this with pip
# import os

# app.config["SECRET_KEY"] = "aaa"  # Change this to a random secret key
# app.config["SESSION_TYPE"] = "filesystem"
# Session(app)

# if not os.path.exists(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():

#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '' or not allowed_file(file.filename):
#             return redirect(request.url)


#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         filename = secure_filename(f"{timestamp}_{file.filename}")
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         # Process the CSV file
#         df = pd.read_csv(filepath,index_col=False)
#         session['progress'] = 10
#         df_vis = get_df_vis(df,3)
#         plot_url = visualize_df_dis(df_vis)
#         plot_url2 = visualize_df_vib(df_vis)
#         # Here you would preprocess your data and predict
#         processed_data = process_data(df)
#         prediction_result = predict_classes(processed_data)
#         session['progress'] = 50
#         # prediction_result = "Class X" # Placeholder for prediction result
#         class_number = 3  # Placeholder for demonstration
#         image_file = f'class{class_number}.png'
#         image_path = url_for('static', filename=image_file)
#         session['progress'] = 100
        
#         return render_template('index.html', plot_url=plot_url, plot_url2=plot_url2, prediction_result=prediction_result, image_path=image_path, step="completed", progress=session['progress'])


# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '' or not allowed_file(file.filename):
#             return redirect(request.url)

#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         filename = secure_filename(f"{timestamp}_{file.filename}")
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # Process the CSV file
#         df = pd.read_csv(filepath, index_col=False)
#         df_vis = get_df_vis(df, 3)
#         plot_url = visualize_df_dis(df_vis)
#         plot_url2 = visualize_df_vib(df_vis)
#         processed_data = process_data(df)
#         prediction_result = predict_classes(processed_data)
#         class_number = 3  # Placeholder for demonstration
#         image_file = f'class{class_number}.png'
#         image_path = url_for('static', filename=image_file)
        
#         # Redirect to the same URL but with a query parameter indicating processing is completed
#         return redirect(url_for('upload_file', processed=1))

#     processed = request.args.get('processed')
#     return render_template('index.html', processed=processed)
#     # Reset progress on GET
#     session['progress'] = 0
#     return render_template('index.html', plot_url=None, prediction_result=None, progress=session['progress'])


# if __name__ == '__main__':
#     app.run(debug=True)


#================================working version for loading.gif
# from flask import Flask, request, render_template, send_from_directory, session, redirect, url_for
# from flask_session import Session
# import os, pandas as pd
# from datetime import datetime
# from werkzeug.utils import secure_filename
# # Assuming predict.py contains necessary functions
# from predict import get_df_vis, visualize_df_dis, visualize_df_vib, process_data, predict_classes

# app = Flask(__name__)

# app.config['UPLOAD_FOLDER'] = 'uploads/'
# app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt'}  # Added 'txt' for example
# app.config["SECRET_KEY"] = "aaa"
# app.config["SESSION_TYPE"] = "filesystem"
# Session(app)

# if not os.path.exists(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     # Initialize variables to None to avoid reference errors in template
#     plot_url = None
#     plot_url2 = None
#     prediction_result = None
#     image_path = None

#     if request.method == 'POST':
#         # File upload and processing logic
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '' or not allowed_file(file.filename):
#             return redirect(request.url)

#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         filename = secure_filename(f"{timestamp}_{file.filename}")
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # Process the CSV file
#         df = pd.read_csv(filepath, index_col=False)
#         df_vis = get_df_vis(df, 3)
#         plot_url = visualize_df_dis(df_vis)
#         plot_url2 = visualize_df_vib(df_vis)
#         processed_data = process_data(df)
#         prediction_result = predict_classes(processed_data)
#         class_number = 3  # Placeholder for demonstration
#         image_file = f'class{class_number}.png'
#         image_path = url_for('static', filename=image_file)

#         # Store results in session or another persistent storage to retrieve after redirect
#         session['plot_url'] = plot_url
#         session['plot_url2'] = plot_url2
#         session['prediction_result'] = prediction_result
#         session['image_path'] = image_path

#         # Redirect with processed indicator
#         return redirect(url_for('upload_file', processed=1))

#     processed = request.args.get('processed')
#     if processed:
#         # Retrieve results from session or persistent storage
#         plot_url = session.get('plot_url')
#         plot_url2 = session.get('plot_url2')
#         prediction_result = session.get('prediction_result')
#         image_path = session.get('image_path')

#     return render_template('index.html', processed=processed, plot_url=plot_url, plot_url2=plot_url2, prediction_result=prediction_result, image_path=image_path)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template, jsonify, url_for, redirect, session
from flask_session import Session
import os, pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
from predict import get_df_vis, visualize_df_dis, visualize_df_vib, process_data, predict_classes

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config["SECRET_KEY"] = "aaa"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    # Initial page load
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'No selected file or file type not allowed'})

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = secure_filename(f"{timestamp}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process the CSV file
    df = pd.read_csv(filepath, index_col=False)
    df_vis = get_df_vis(df, 3)
    plot_url = visualize_df_dis(df_vis)
    plot_url2 = visualize_df_vib(df_vis)
    processed_data = process_data(df)
    prediction_result = predict_classes(processed_data)
    
    # Assuming visualize_df_dis and visualize_df_vib return base64 encoded images
    # And predict_classes returns a string result

    return jsonify({
        'plot_url': plot_url,
        'plot_url2': plot_url2,
        'prediction_result': prediction_result
    })

if __name__ == '__main__':
    app.run(debug=True)
