import sys
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from yolo import YOLO

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Object Detection")
        self.setGeometry(100, 100, 800, 600)
        
        self.yolo = YOLO()
        
        self.initUI()
    
    def initUI(self):
        self.layout = QVBoxLayout()

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.btn_predict = QPushButton("Predict Image", self)
        self.btn_predict.clicked.connect(self.predict_image)
        self.layout.addWidget(self.btn_predict)

        self.btn_video = QPushButton("Detect Video", self)
        self.btn_video.clicked.connect(self.detect_video)
        self.layout.addWidget(self.btn_video)

        self.btn_camera = QPushButton("Capture from Camera", self)
        self.btn_camera.clicked.connect(self.capture_from_camera)
        self.layout.addWidget(self.btn_camera)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def predict_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_name:
            image = Image.open(file_name)
            r_image = self.yolo.detect_image(image)
            r_image.show()
            self.display_image(r_image)

    def detect_video(self):
        # Video detection function (unchanged)
        pass

    def capture_from_camera(self):
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            raise ValueError("Error opening camera")

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = np.array(self.yolo.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()

    def display_image(self, image):
        image = image.convert("RGB")
        image = np.array(image)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.label.size(), Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
