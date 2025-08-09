from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # model nhỏ

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    output_image = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            results = model.predict(source=filepath, conf=0.25, classes=0)  # chỉ phát hiện người
            r = results[0]
            annotated = r.plot()
            out_path = os.path.join(UPLOAD_FOLDER, "result_" + file.filename)
            cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            output_image = out_path
    return render_template("index.html", output_image=output_image)

if __name__ == "__main__":
    app.run(debug=True)
