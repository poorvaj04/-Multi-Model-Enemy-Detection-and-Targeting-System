from flask import Flask, render_template, request
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load models once
drone_model = YOLO('runs/bestd.pt')
general_model = YOLO('runs/bestg.pt')
terrorist_model = YOLO('runs/bestt.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    output_video = None

    if request.method == 'POST':
        file = request.files['video']
        if file:
            input_path = os.path.join('static', 'input.mp4')
            output_path = os.path.join('static', 'output_video_terr.mp4')
            file.save(input_path)

            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                for model, color in zip([drone_model, general_model, terrorist_model],
                                        [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
                    results = model(frame)
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = box.conf.item()
                            cls = int(box.cls.item())
                            label = model.names[cls]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                out.write(frame)

            cap.release()
            out.release()
            output_video = 'output_video_terr.mp4'

    return render_template('index.html', output_video=output_video)

if __name__ == '__main__':
    app.run(debug=True)