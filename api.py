from flask import Flask, request, jsonify, send_file
from LiidClassifierModel import LiidClassifierModel
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
model = LiidClassifierModel()

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed extensions
ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/train', methods=['POST'])
def train():
    try:
        if 'dataset' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['dataset']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            result = model.train(file_path)
            return jsonify({'message': result}), 200
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'dataset' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['dataset']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            result = model.predict(file_path)
            return send_file('data/results.csv', as_attachment=True), 200
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/test', methods=['get'])
def test():
    return jsonify("connected"), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
