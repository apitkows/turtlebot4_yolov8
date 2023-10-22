#!/usr/bin/env python3

# Importieren der erforderlichen Bibliotheken
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Pfad zum YOLO-Modell als Konstante
YOLO_MODEL_PATH = '/home/ubuntu/python/src/topic/yolov8_fire-extinguisher.pt'

# Hauptklasse für die Bildverarbeitung und Steuerung
class ImageDisplayNode(Node):
    def __init__(self):
        super().__init__('image_display_node')
        
        # Initialisierung der Variablen
        self.bridge = CvBridge()  # Für die Konvertierung zwischen ROS- und OpenCV-Bildern
        self.model = YOLO(YOLO_MODEL_PATH)  # YOLO-Modell für die Objekterkennung
        self.counter = 0  # FPS-Zähler
        self.startTime = time.monotonic()  # Startzeit für FPS-Berechnung

        # Initialisierung der ROS-Kommunikation
        self.qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.subscription = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.image_callback,
            self.qos_policy
        )

    # Callback-Funktion für die Bildverarbeitung
    def image_callback(self, msg):
        # Konvertieren des ROS-Bilds in ein OpenCV-Bild
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Skalieren des Bilds auf 640x640 Pixel
        resized_image = cv2.resize(cv_image, (640, 640))
        
        # Anwenden des YOLO-Modells zur Objekterkennung
        results = self.model.predict(source=resized_image)

        # Annotieren und Anzeigen des Bilds
        for r in results:
            annotator = Annotator(resized_image)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])
        
        resized_image = annotator.result()
        
        # FPS-Berechnung und Anzeige
        self.counter += 1
        cv2.putText(resized_image, "YOLOV8 FPS: {:.2f}".format(self.counter / (time.monotonic() - self.startTime)),
                    (2, resized_image.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 1, (50,205,50))
        
        cv2.imshow('TB4 CAM', resized_image)
        
        # Beenden der Anwendung bei Drücken der Taste 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()

# Hauptfunktion
def main():
    rclpy.init()
    node = ImageDisplayNode()
    rclpy.spin(node)

# Startpunkt des Programms
if __name__ == '__main__':
    main()
