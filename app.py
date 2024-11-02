import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO(r'C:\Users\meggs\runs\detect\train9\weights\best.pt')
upload_folder = r'static/uploads'
output_folder = r'static/outputs'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html') #Display index.html

def upload_file():
    if 'file' not in request.files:
        return jsonify({'error' : 'No file part in request'}), 400
    
    file = request.files['file']

    if file.filename == ' ':
        return jsonify({'error' : 'No selection file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        results = model(file_path)

        output_path = os.path.join(output_folder, filename)
        for result in results:
            result.save(filename = output_path)

        return jsonify({'image path': output_path})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/outputs/<filename>')
def send_output(filename):
    return send_from_directory(output_folder, filename)

@app.route('/get_media', methods=['POST'])
def get_media():
    prompt = request.json.get('prompt').lower()
    response = {}