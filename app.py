
from flask import Flask, request, render_template, jsonify, url_for, redirect, session
from flask_session import Session
import os, pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
from predict import *

from flask import Flask, request, jsonify, render_template, url_for, redirect, session
from werkzeug.utils import secure_filename
import os, pandas as pd
from datetime import datetime
# Assuming necessary import statements for your prediction and visualization functions are here

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config["SECRET_KEY"] = "aaa"
app.config["SESSION_TYPE"] = "filesystem"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_visualize', methods=['POST'])
def upload_visualize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'No selected file or file type not allowed'})
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = secure_filename(f"{timestamp}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    session['last_uploaded_filepath'] = filepath
    # Process for visualization
    df = pd.read_csv(filepath,index_col=False)
    # Assuming get_df_vis, visualize_df_dis, and visualize_df_vib return base64 encoded images
    df_vis = get_df_vis(df, 3)
    plot_url = visualize_df_dis(df_vis)
    plot_url2 = visualize_df_vib(df_vis)
    
    return jsonify({
        'plot_url': plot_url,
        'plot_url2': plot_url2,
    })

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Assuming the filename or data to predict on is passed as part of the request
#     # Or use session or another mechanism to store/retrieve the filepath or data
#     filepath = session.get('last_uploaded_filepath') # Example, adjust based on your implementation
    
#     # Perform prediction
#     df = pd.read_csv(filepath,index_col=False)
#     processed_data = process_data(df)
#     prediction_result = predict_classes(processed_data)
    
#     # Example usage
#     folder_path = "Failure Mode Images"
#     start_number = prediction_result  # Example start number

#     # Fetch the first base64 encoded image that starts with the specified number
#     encoded_image = get_images_from_folder(folder_path, start_number)
#     return jsonify({'prediction_result': prediction_result,'plot_url3':encoded_image})
@app.route('/predict', methods=['POST'])
def predict():
    filepath = session.get('last_uploaded_filepath')
    df = pd.read_csv(filepath, index_col=False)
    processed_data = process_data(df)
    prediction_result,shap_plot = predictX(processed_data)

    # Assuming the images are named with a number prefix and are stored in 'static' folder
    folder_path = os.path.join(os.getcwd(), "Failure Mode Images")
    files = os.listdir(folder_path)
    files.sort()  # Ensure files are sorted correctly, might need custom sorting for numeric prefixes
    
    # Select the image based on prediction_result
    selected_image = None
    for file in files:
        if file.startswith(str(prediction_result) + " "):  # Match files starting with prediction_result
            selected_image = file
            break
    
    if selected_image:
        image_path = url_for('static', filename=selected_image)
        # Extract the descriptive text from the filename
        
        descriptive_text = extract_description(selected_image) # Removes number prefix and file extension
    else:
        image_path = ""
        descriptive_text = "No image found"

    return jsonify({
        'prediction_result': prediction_result,
        'image_path': image_path,
        'descriptive_text': descriptive_text,
        'shap_plot':url_for('static',filename=shap_plot)
    })
if __name__ == '__main__':
    app.run(debug=True)
