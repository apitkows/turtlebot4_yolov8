#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
from rclpy.executors import MultiThreadedExecutor
import threading

import cv2
import depthai as dai
import numpy as np
import time

# Konfigurationskonstanten
FRONT_DISTANCE_THRESHOLD = 0.7
TURN_ANGLE = math.pi / 4
MAX_TURN_COUNT = 100
NN_CONFIDENCE_THRESHOLD = 0.9
NN_NUM_CLASSES = 1
NN_IOU_THRESHOLD = 0.5
NN_NUM_INFERENCE_THREADS = 2
NN_PATH = "/home/ubuntu/python/src/oakd/yolov8n_fire-extinguisher.blob"

class RobotExplorer(Node):
    def __init__(self):
        super().__init__('robot_explorer')
        # Initialisierung der Zustands- und Bewegungsvariablen
        self.state = "EXPLORER"
        self.twist = Twist()
        self.turn_count = 0
        self.turn_direction = 1

        # QoS-Policy-Einstellungen
        self.qos_policy = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Abonnement für LaserScan-Daten
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            self.qos_policy
        )

        # Publisher für Bewegungsbefehle
        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

    def listener_callback(self, msg):
        """Callback für LaserScan-Daten."""
        # Wenn der Roboter nicht im Erkundungsmodus ist, tue nichts
        if self.state != "EXPLORER":
            return

        # Extrahieren der Winkelinformationen aus dem LaserScan
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment

        # Berechnung des Index für die Front des Roboters
        index_front = round((-1.5708 - angle_min) / angle_increment)

        # Überprüfen, ob ein Hindernis im 50-Grad-Bereich vor dem Roboter ist
        index_start = max(0, index_front - 50)
        index_end = min(len(msg.ranges) - 1, index_front + 50)
        front_ranges = msg.ranges[index_start:index_end+1]
        front_distance = min([x for x in front_ranges if x != float('inf')])

        # Entscheidungslogik für die Bewegung des Roboters
        if front_distance < FRONT_DISTANCE_THRESHOLD:
            # Ein Hindernis ist vor dem Roboter, drehen Sie ihn um 45 Grad
            self.twist.linear.x = 0.0
            self.twist.angular.z = TURN_ANGLE * self.turn_direction
            
            # Aktualisieren des Drehzählers
            self.turn_count += 1
            
            # Wechsel der Drehrichtung nach 100 Drehungen
            if self.turn_count >= MAX_TURN_COUNT:
                self.turn_direction *= -1
                self.turn_count = 0
        else:
            # Kein Hindernis, fahren Sie geradeaus
            self.twist.linear.x = 0.2
            self.twist.angular.z = 0.0

        # Veröffentlichen der Bewegungsbefehle
        self.publisher.publish(self.twist)

class RobotAlligner(Node):
    def __init__(self):
        super().__init__('robot_alligner')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def allign_robot(self, box, image_width=640):
        """Richtet den Roboter anhand der Bounding-Box aus."""
        # Berechnung des Zentrums der Bounding-Box
        x_center = (box[0] + box[2]) / 2
        error = x_center - image_width / 2
        area = abs(box[2] - box[0]) * abs(box[3] - box[1])
        twist = Twist()
        # Wenn der Fehler größer als 75 ist, drehen Sie den Roboter
        if abs(error) > 75:
            twist.angular.z = -0.1 if error > 0 else 0.1
        # Wenn die Fläche der Bounding-Box kleiner als 70000 ist, fahren Sie vorwärts
        elif area < 70000:
            twist.linear.x = 0.1
        self.publisher.publish(twist)

class ObjectDetection(Node):
    def __init__(self, robotExplorer, robotAlligner):
        super().__init__('object_detection')
        self.robotExplorer = robotExplorer
        self.robotAlligner = robotAlligner
        self.labels = ["fire_extinguisher"]
        self.syncNN = True

    def run(self):
        """Hauptmethode für die Objekterkennung und Robotersteuerung."""
        # Erstellen der Pipeline für die Kamera und das neuronale Netzwerk
        pipeline = dai.Pipeline()

        # Definieren der Quellen und Ausgänge
        camRgb = pipeline.create(dai.node.ColorCamera)
        detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        nnOut = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")
        nnOut.setStreamName("nn")

        # Kameraeigenschaften
        camRgb.setPreviewSize(640, 640)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        camRgb.setFps(30)

        # Netzwerkeigenschaften
        detectionNetwork.setConfidenceThreshold(NN_CONFIDENCE_THRESHOLD)
        detectionNetwork.setNumClasses(NN_NUM_CLASSES)
        detectionNetwork.setCoordinateSize(4)
        detectionNetwork.setAnchors([])
        detectionNetwork.setAnchorMasks({})
        detectionNetwork.setIouThreshold(NN_IOU_THRESHOLD)
        detectionNetwork.setBlobPath(NN_PATH)
        detectionNetwork.setNumInferenceThreads(NN_NUM_INFERENCE_THREADS)
        detectionNetwork.input.setBlocking(False)

        # Verknüpfung der Knoten
        camRgb.preview.link(detectionNetwork.input)
        if self.syncNN:
            detectionNetwork.passthrough.link(xoutRgb.input)
        else:
            camRgb.preview.link(xoutRgb.input)
        detectionNetwork.out.link(nnOut.input)

        # Verbindung zum Gerät und Start der Pipeline
        with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            frame = None
            detections = []
            startTime = time.monotonic()
            counter = 0

            # Funktionen für die Frame-Normalisierung und -Anzeige
            def frameNorm(frame, bbox):
                normVals = np.full(len(bbox), frame.shape[0])
                normVals[::2] = frame.shape[1]
                return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

            def displayFrame(name, frame):
                if len(detections) > 0:
                    self.robotExplorer.state = "ALLIGN"
                else:
                    self.robotExplorer.state = 'EXPLORER'

                for detection in detections:
                    bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.putText(frame, self.labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    self.robotAlligner.allign_robot(bbox)
                cv2.imshow(name, frame)

            # Hauptverarbeitungsschleife
            while True:
                if self.syncNN:
                    inRgb = qRgb.get()
                    inDet = qDet.get()
                else:
                    inRgb = qRgb.tryGet()
                    inDet = qDet.tryGet()

                if inRgb is not None:
                    frame = inRgb.getCvFrame()
                    
                    cv2.putText(frame, "YOLOV8 FPS: {:.2f}".format(counter / (time.monotonic() - startTime)),
                    (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 1, (50,205,50))

                if inDet is not None:
                    detections = inDet.detections
                    counter += 1

                if frame is not None:
                    displayFrame("rgb", frame)

                if cv2.waitKey(1) == ord('q'):
                    break

def main():
    rclpy.init()
    robotExplorer = RobotExplorer()
    robotAlligner = RobotAlligner()
    objectDetection = ObjectDetection(robotExplorer, robotAlligner)

    executor = MultiThreadedExecutor()
    executor.add_node(robotExplorer)
    thread = threading.Thread(target=executor.spin)
    thread.start()

    try:
        objectDetection.run()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
