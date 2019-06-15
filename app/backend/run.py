from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from config import DEBUG, BACKEND_HOST, BACKEND_HOST_PORT, IMAGE_FOLDER, VOCAB_PATH, CHECKPOINT_PATHS
from sample import get_sample, Vocabulary
import os

if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

app = Flask(__name__, static_url_path='')
app.config.from_object(__name__)

CORS(app, resources={r'/*': {'origins': '*'}})


@app.route('/images/<path:filename>')
def serve_img(filename):
    img_folder = IMAGE_FOLDER.replace('/', '')
    img_folder = img_folder.replace('.', '')
    return send_from_directory(img_folder, filename)


@app.route('/generate', methods=['POST'])
def generate():
    mode = request.args.get('mode')
    modes = ['factual', 'happy', 'sad', 'angry']

    # check file upload and mode params
    if 'file' not in request.files or mode not in modes:
        return jsonify({
            'nic': '-',
            'nic_att': '-',
            'stylenet': '-',
            'stylenet_att': '-',
            'path_img': '-'
        })

    image_file = request.files['file']
    path = IMAGE_FOLDER + image_file.filename
    try:
        image_file.save(path)
        result = {
            'nic':
            get_sample(CHECKPOINT_PATHS['nic'][mode], VOCAB_PATH, mode, False,
                       path),
            'nic_att':
            get_sample(CHECKPOINT_PATHS['nic_att'][mode], VOCAB_PATH, mode,
                       True, path),
            'stylenet':
            get_sample(CHECKPOINT_PATHS['stylenet'][mode], VOCAB_PATH, mode,
                       False, path),
            'stylenet_att':
            get_sample(CHECKPOINT_PATHS['stylenet_att'][mode], VOCAB_PATH, mode,
                       True, path),
            'path_img':
            '/images/' + image_file.filename
        }
    except Exception as e:
        return str(e), 500

    return jsonify(result)


if __name__ == '__main__':
    app.run(host=BACKEND_HOST, port=BACKEND_HOST_PORT, debug=DEBUG)
