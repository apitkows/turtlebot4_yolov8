#!/usr/bin/env python3

# Importieren der erforderlichen Bibliotheken
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
import math
import cv2
from cv_bridge import CvBridge
import time
from ultralytics import YOLO
from rclpy.executors import MultiThreadedExecutor
import threading

# Definieren von Konstanten für die Robotersteuerung
MIN_FRONT_DISTANCE = 0.7
LINEAR_SPEED = 0.2
ANGULAR_SPEED = math.pi / 4
TURN_COUNT_THRESHOLD = 100
YOLO_MODEL_PATH = '/home/ubuntu/python/src/topic/yolov8n_fire-extinguisher.pt'

# Klasse für den Erkundungsmodus des Roboters
class RobotExplorer(Node):
    def __init__(self):
        super().__init__('robot_explorer')
        self.init_variables()
        self.init_ros_communication()

    # Initialisierung der Variablen
    def init_variables(self):
        self.state = "EXPLORER"  # Setzt den Anfangszustand auf "EXPLORER"
        self.twist = Twist()  # Initialisiert die Twist-Nachricht für die Bewegungsbefehle
        self.turn_count = 0  # Zählt die Anzahl der Drehungen
        self.turn_direction = 1  # Startet mit Drehung nach rechts (1 für rechts, -1 für links)

    # Initialisierung der ROS-Kommunikation
    def init_ros_communication(self):
        # Einstellen der QoS-Policy
        self.qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Erstellen eines Abonnements für den LaserScan-Topic
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            self.qos_policy
        )
        # Erstellen eines Publishers für den cmd_vel-Topic
        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

    # Callback-Funktion für den LaserScan
    def listener_callback(self, msg):
        if self.state != "EXPLORER":
            return
        self.process_laser_scan(msg)

    # Verarbeitung der LaserScan-Daten
    def process_laser_scan(self, msg):
        # Extrahieren der Winkelinformationen aus dem LaserScan
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment

        # Berechnung des Index für die Front des Roboters
        index_front = round((-1.5708 - angle_min) / angle_increment)  # -1.5708 rad entspricht -90 Grad

        # Überprüfen, ob ein Hindernis im 50-Grad-Bereich vor dem Roboter ist
        index_start = max(0, index_front - 50)
        index_end = min(len(msg.ranges) - 1, index_front + 50)
        front_ranges = msg.ranges[index_start:index_end+1]
        front_distance = min([x for x in front_ranges if x != float('inf')])

        # Entscheidungslogik für die Bewegung des Roboters
        if front_distance < MIN_FRONT_DISTANCE:
            # Ein Hindernis ist vor dem Roboter, drehen Sie ihn um 45 Grad
            self.twist.linear.x = 0.0
            self.twist.angular.z = ANGULAR_SPEED * self.turn_direction  # 45 Grad in Rad
            
            # Aktualisieren des Drehzählers
            self.turn_count += 1
            
            # Wechsel der Drehrichtung nach 100 Drehungen
            if self.turn_count >= TURN_COUNT_THRESHOLD:
                self.turn_direction *= -1
                self.turn_count = 0
        else:
            # Kein Hindernis, fahren Sie geradeaus
            self.twist.linear.x = LINEAR_SPEED
            self.twist.angular.z = 0.0

        # Veröffentlichen der Bewegungsbefehle
        self.publisher.publish(self.twist)

# Klasse für die Ausrichtung des Roboters auf ein Ziel
class RobotAlligner(Node):
    def __init__(self):
        super().__init__('robot_alligner')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    # Funktion zur Ausrichtung des Roboters
    def allign_robot(self, x_min, y_min, x_max, y_max, image_width=640):
        # Berechnung des Zentrums des erkannten Objekts
        x_center = (x_min + x_max) / 2
        # Berechnung des Fehlers zwischen dem Zentrum des Objekts und dem Zentrum des Bildes
        error = x_center - image_width / 2
        # Berechnung der Fläche des erkannten Objekts
        area = abs(x_max - x_min) * abs(y_max - y_min)
        
        twist = Twist()
        # Wenn der Fehler groß ist, drehen Sie den Roboter, um das Objekt zu zentrieren
        if abs(error) > 150:
            twist.angular.z = -0.1 if error > 0 else 0.1
        # Wenn die Fläche des Objekts klein ist, nähern Sie sich dem Objekt
        elif area < 70000:
            twist.linear.x = 0.1
        # Veröffentlichen der Bewegungsbefehle
        self.publisher.publish(twist)

# Hauptklasse für die Bildverarbeitung und Steuerung
class ImageDisplayNode(Node):
    def __init__(self):
        super().__init__('image_display_node')
        self.init_variables()
        self.init_ros_communication()
        self.counter = 0  # FPS-Zähler
        self.startTime = time.monotonic()  # Startzeit für FPS-Berechnung

    # Initialisierung der Variablen
    def init_variables(self):
        self.bridge = CvBridge()  # Für die Konvertierung zwischen ROS- und OpenCV-Bildern
        self.model = YOLO(YOLO_MODEL_PATH)  # YOLO-Modell für die Objekterkennung
        self.robot_explorer = RobotExplorer()  # Instanz der RobotExplorer-Klasse
        self.robot_alligner = RobotAlligner()  # Instanz der RobotAlligner-Klasse

    # Initialisierung der ROS-Kommunikation
    def init_ros_communication(self):
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
        # Erstellen eines MultiThreadedExecutors für den Explorer-Node
        executor = MultiThreadedExecutor()
        executor.add_node(self.robot_explorer)
        thread = threading.Thread(target=executor.spin)
        thread.start()

    # Callback-Funktion für die Bildverarbeitung
    def image_callback(self, msg):
        # Konvertieren des ROS-Bilds in ein OpenCV-Bild
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Skalieren des Bilds auf 640x640 Pixel
        resized_image = cv2.resize(cv_image, (640, 640))
        # Anwenden des YOLO-Modells zur Objekterkennung
        results = self.model.predict(source=resized_image)

        # Durchlaufen der erkannten Objekte und Entscheidung für die Roboteraktion
        for r in results:
            boxes = r.boxes
            box = None
            for b in boxes:
                if b.conf > 0.9:
                    box = b
                    break
            if box is not None:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                self.robot_explorer.state = "ALLIGN"
                self.robot_alligner.allign_robot(x_min, y_min, x_max, y_max) 
            else:
                self.robot_explorer.state = "EXPLORER"

        # Annotieren und Anzeigen des Bilds
        annotated_frame = results[0].plot()
        self.counter += 1
        # FPS-Berechnung und Anzeige
        cv2.putText(annotated_frame, "YOLOV8 FPS: {:.2f}".format(self.counter / (time.monotonic() - self.startTime)),
                    (2, annotated_frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 1, (50,205,50))
        
        cv2.imshow('TB4 CAM', annotated_frame)
        
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

