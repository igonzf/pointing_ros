
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
    
    
    def is_ray_bbox_intersection(self, p1, p2, bbox) -> bool:
        ''' Check if the 3D line defined by two points intersects a 3D bounding box.
            Arguments:
                p1 {geometry_msgs/Point}: First point defining the line.
                p2 {geometry_msgs/Point}: Second point defining the line.
                bbox {yolov8_msgs/BoundingBox3D}: Bounding box.
            Return:
                intersection {bool}:
                    True if the line intersects the bounding box, False otherwise.
        '''
        # Extract coordinates of keypoints
        p1_coords = np.array([p1.x, p1.y, p1.z])
        p2_coords = np.array([p2.x, p2.y, p2.z])
        
        # Calculate direction vector of the line
        direction = p2_coords - p1_coords
        direction /= np.linalg.norm(direction)
        
        # Extract center and size of bounding box
        center = bbox.center.position
        size = bbox.size
        
        # Calculate half size
        half_size = np.array([size.x / 2.0, size.y / 2.0, size.z / 2.0])
        
        # Calculate min and max corners of the bounding box
        min_corner = np.array([center.x, center.y, center.z]) - half_size
        max_corner = np.array([center.x, center.y, center.z]) + half_size
        
        # Calculate distances to each face of the bounding box
        t_min = (min_corner - p1_coords) / direction
        t_max = (max_corner - p1_coords) / direction
        
        # Use broadcasting to find the minimum and maximum of t values
        t_enter = np.min(np.maximum(t_min, t_max))
        t_exit = np.max(np.minimum(t_min, t_max))
        
        # Check if the line intersects the bounding box
        return t_enter < t_exit and t_exit > 0
    
    

    
    def is_bbox_intersection(self, m_p1, m_p2, bbox) -> bool:
        p1 = np.array([m_p1.x, m_p1.y, m_p1.z])
        p2 = np.array([m_p2.x, m_p2.y, m_p2.z])


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
                    

                    # Check if the person is pointing
                    if is_pointing:
                        self.get_logger().info("PERSON IS POINTING")
                        self.get_logger().info(str(right_arm_xyz[0]))
                        self.get_logger().info(str(right_arm_xyz[1]))

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
                                
                            self._v_markers_pub.publish(v_markers)

                            bbox_faces = calculate_bbox_faces(bbox_v)
                            p1 = np.array([marker.points[0].x, marker.points[0].y, marker.points[0].z])
                            p2 = np.array([marker.points[1].x, marker.points[1].y, marker.points[1].z])
                            if is_intersection(p1, p2, bbox_faces):
                                self.get_logger().info(f'PERSON POINTED TO THE BAG ID: {detection.id}')
                            """ if self.is_bbox_intersection(marker.points[0], marker.points[1], bbox):
                                self.get_logger().info(f'PERSON POINTED TO THE BAG ID: {detection.id}') """
                        """ for detection in msg_object_detections.detections:

                            bbox = detection.bbox3d
                            if self.is_ray_bbox_intersection(right_arm[0].point, right_arm[2].point, bbox):
                                self.get_logger().info(f'PERSON POINTED TO THE BAG ID: {detection.id}')
                                pointing_msg.object = bbox
                                self._pointing_pub.publish(pointing_msg) """
                        
                        self._markers_pub.publish(direction_marker_array)


def main():
    rclpy.init()
    node = PointingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
