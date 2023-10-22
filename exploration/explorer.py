#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class RobotExplorer(Node):
    def __init__(self):
        super().__init__('robot_explorer')

        self.state = "EXPLORER"

        # Initialisierung der Variablen
        self.twist = Twist()
        self.turn_count = 0  # Zählt die Anzahl der Drehungen
        self.turn_direction = 1  # Startet mit Drehung nach rechts (1 für rechts, -1 für links)

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
            10  # Standard-QoS
        )

    # Callback-Funktion, die bei jedem neuen LaserScan aufgerufen wird
    def listener_callback(self, msg):

        if self.state != "EXPLORER":
            return

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

        # Logging der Frontdistanz
        self.get_logger().info(f'Front distance: {front_distance}')

        # Entscheidungslogik für die Bewegung des Roboters
        if front_distance < 0.7:
            # Ein Hindernis ist vor dem Roboter, drehen Sie ihn um 45 Grad
            self.twist.linear.x = 0.0
            self.twist.angular.z = math.pi / 4 * self.turn_direction  # 45 Grad in Rad
            
            # Aktualisieren des Drehzählers
            self.turn_count += 1
            
            # Wechsel der Drehrichtung nach 100 Drehungen
            if self.turn_count >= 100:
                self.turn_direction *= -1
                self.turn_count = 0
        else:
            # Kein Hindernis, fahren Sie geradeaus
            self.twist.linear.x = 0.2
            self.twist.angular.z = 0.0

        # Veröffentlichen der Bewegungsbefehle
        self.publisher.publish(self.twist)

def main():
    rclpy.init()
    robotExplorer =  RobotExplorer()
    rclpy.spin(robotExplorer)

if __name__ == '__main__':
    main()