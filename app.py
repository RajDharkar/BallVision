import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Model and directory configuration
model_path = r'C:\Users\meggs\runs\detect\train9\weights\best.pt'
model = YOLO(model_path)

upload_folder = 'static/uploads'
output_folder = 'static/outputs'
app.config['UPLOAD_FOLDER'] = upload_folder
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Allowed file types
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Check if file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        print("File uploaded and saved:", file_path)

        # Process the image
        results = model(file_path)

        output_path = os.path.join(output_folder, filename)
        for result in results:
            result.save(filename=output_path)

        url_path = f'outputs/{filename}'

        print("Processed image saved to:", output_path)

        return jsonify({'image_path': url_path})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/outputs/<filename>')
def send_output(filename):
    return send_from_directory(output_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
