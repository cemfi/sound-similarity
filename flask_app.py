import os
from flask import Flask, render_template, request, send_from_directory
import hashlib
import analyze_sound

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=['POST'])
def upload():
    target = 'audio/'
    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        _, ext = os.path.splitext(file.filename)
        myfilename = md5(file)
        destination = os.path.join(target, myfilename) + ext
        file.save(destination)
        file.close()
        result_path = analyze_sound.process(destination, method='tsne', features='normal')

    return render_template("visualization.html", json_file=result_path)


@app.route('/audio/<path:path>')
def send_audio(path):
    return send_from_directory('audio', path)


@app.route('/results/<path:path>')
def send_result(path):
    return send_from_directory('results', path)


def md5(f):
    hash_md5 = hashlib.md5()
    for chunk in iter(lambda: f.read(4096), b''):
        hash_md5.update(chunk)
    f.seek(0)
    return hash_md5.hexdigest()


if __name__ == "__main__":
    app.run(port=4555)
