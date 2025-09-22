import os
from flask import request, render_template, current_app
from werkzeug.utils import secure_filename
from flask import send_from_directory

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def register_routes(app, predictor):
    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/')
    def index():
        return render_template('index.html', prediction_text="Upload an image to get a prediction.")

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return render_template('index.html', prediction_text='No file part in the request.')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', prediction_text='No file selected for uploading.')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_folder = app.config['UPLOAD_FOLDER']
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            prediction_result = predictor.predict(filepath)
            image_url = url_for('uploaded_file', filename=filename)
            return render_template('index.html', prediction_text=prediction_result, image_url=image_url)
        else:
            return render_template('index.html', prediction_text='Allowed file types are png, jpg, jpeg, gif.')
