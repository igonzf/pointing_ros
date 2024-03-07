
import math

import numpy as np
from array import *
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import MarkerArray
from pointing_msgs.msg import Pointing
from yolov8_msgs.msg import KeyPoint3DArray
from yolov8_msgs.msg import DetectionArray
from yolov8_msgs.msg import BoundingBox3D
from .rviz_makers import create_3d_pointing_ray, create_marker_point
from .calculate_functions import calculate_bbox_v, calculate_bbox_faces, is_intersection

from message_filters import Subscriber, ApproximateTimeSynchronizer

class PointingNode(Node):

    def __init__(self) -> None:
        super().__init__("pointing_node")

        self.right_arm_index = [6, 8, 10]
        self.left_arm_index = [5, 7, 9]
        self.right_arm_xyz = []

        self._class_to_color = {}
        self.cv_bridge = CvBridge()

        # topics
        self._markers_pub = self.create_publisher(MarkerArray, "markers", 10)
        self._v_markers_pub = self.create_publisher(MarkerArray, "v_markers", 10)
        self._pointing_pub = self.create_publisher(Pointing, "pointing", 10)
        

        self._detections_3d_sub = Subscriber(
            self, DetectionArray, "detections_3d"
        )
        self._object_detections_sub = Subscriber(
            self, DetectionArray, "object_detections_3d"
        )
        self._cam_info_sub = Subscriber(
            self, CameraInfo, "camera_info", qos_profile= qos_profile_sensor_data
        )
        
        tss = ApproximateTimeSynchronizer([ self._object_detections_sub, self._detections_3d_sub], 30, 0.1)
        tss.registerCallback(self.pointing_cb)

    def get_right_arm(self, keypoint_array) -> KeyPoint3DArray:
        right_arm = []
        kpts_right_arm = [i for i in keypoint_array if i.id in self.right_arm_index]

        if len(kpts_right_arm)>2:            
            right_arm = [i for i in kpts_right_arm if i.point.x != 0 and i.point.y != 0]

        return right_arm
    
    def is_arm_stretched(self, arm_xyz, angle_thresh=30.0) -> bool:
        p0, p1, p2 = arm_xyz[0], arm_xyz[1], arm_xyz[2]
        vec1 = np.array(p1 - p0)
        vec2 = np.array(p2 - p1)
        angle = np.arccos(
            vec1.dot(vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        angle = angle / math.pi * 180.0
        is_stretched = np.abs(angle) <= angle_thresh
        return is_stretched


    def pointing_cb(self, msg_object_detections: DetectionArray, msg_detections_3d: DetectionArray) -> None:

        if msg_detections_3d.detections:
            human_detections = msg_detections_3d.detections

            for human_pose in human_detections:
                pointing_msg = Pointing()

                right_arm = self.get_right_arm(human_pose.keypoints3d.data)

                # Check if arm exists
                if len(right_arm) == 3:

                    right_arm_xyz = []
                    for k in right_arm:
                        right_arm_xyz.append([k.point.x, k.point.y, k.point.z])

                    is_pointing = self.is_arm_stretched(np.array(right_arm_xyz))
                    

                    # Check if the person is pointing (check 3 points of the arm)
                    if is_pointing:

                        pointing_msg.src.id = int(human_pose.id)
                        pointing_msg.src = human_pose.keypoints3d.data[0]

                        direction_marker_array = MarkerArray()
                        marker = create_3d_pointing_ray('base_link', right_arm_xyz)
                        marker.id = len(direction_marker_array.markers)
                        direction_marker_array.markers.append(marker)
                        
                        
                        
                        # Check if the trajectory of the arm streched intersects a 3D bounding box

                        for detection in msg_object_detections.detections:
                            bbox = detection.bbox3d
                            bbox_v = calculate_bbox_v(bbox)
                            v_markers = MarkerArray()
                            for v in bbox_v:
                                v_marker = create_marker_point('base_link', v[0], v[1], v[2])
                                v_marker.id = len(v_markers.markers)
                                v_markers.markers.append(v_marker)
                                
                            # debug bbox corners
                            self._v_markers_pub.publish(v_markers)

                            bbox_faces = calculate_bbox_faces(bbox_v)
                            p1 = np.array([marker.points[0].x, marker.points[0].y, marker.points[0].z])
                            p2 = np.array([marker.points[1].x, marker.points[1].y, marker.points[1].z])
                            if is_intersection(p1, p2, bbox_faces):
                                self.get_logger().info(f'PERSON POINTED TO THE BAG ID: {detection.id}')
                                
                                pointing_msg = Pointing()
                                pointing_msg.src = human_pose
                                pointing_msg.object = detection
                                self._pointing_pub.publish(pointing_msg)
                        
                        self._markers_pub.publish(direction_marker_array)


def main():
    rclpy.init()
    node = PointingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
