
import math

import numpy as np
from array import *
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from cv_bridge import CvBridge
from rclpy.duration import Duration

from sensor_msgs_py import point_cloud2
#from vision_msgs.msg import Detection2DArray, BoundingBox2D, BoundingBox3D, Detection3D
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import MarkerArray
from pointing_msgs.msg import Pointing
from yolov8_msgs.msg import KeyPoint3DArray
from yolov8_msgs.msg import DetectionArray
from yolov8_msgs.msg import BoundingBox3D

#from yolov8_msgs.msg import PersonKeypoints2D, PersonKeypoints2DArray, Pointing
from geometry_msgs.msg import Point, Pose

from message_filters import Subscriber, ApproximateTimeSynchronizer

from .calc_func import point_plane_distance, point_3d_line_distance, point2dto3d, get_intrinsic_matrix, get_3d_bbox, is_xy_in_bbox
from .rviz_makers import create_marker_point, create_3d_pointing_ray

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
        self._pointing_pub = self.create_publisher(Pointing, "pointing", 10)

        self._detections_3d_sub = Subscriber(
            self, DetectionArray, "detections_3d"
        )
        self._object_detections_sub = Subscriber(
            self, DetectionArray, "object_detections_3d"
        )
        self._pcl_sub = Subscriber(
            self, PointCloud2, "depth_points", qos_profile= qos_profile_sensor_data
        )
        self._cam_info_sub = Subscriber(
            self, CameraInfo, "camera_info", qos_profile= qos_profile_sensor_data
        )
        
        tss = ApproximateTimeSynchronizer([ self._object_detections_sub, self._detections_3d_sub], 30, 0.1)
        tss.registerCallback(self.pointing_cb)

    def get_right_arm(self, keypoint_array) -> KeyPoint3DArray:
        right_arm = []
        #self.get_logger().info(f'TAMAÑO KEYPOINTS DE UNA PERSONA {len(keypoint_array)}')
        kpts_right_arm = [i for i in keypoint_array if i.id in self.right_arm_index]

        if len(kpts_right_arm)>2:
            #kpts_right_arm = [keypoint_array[i] for i in self.right_arm_index]
            
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

    def get_3d_ray_hit_point(self, arm_xyz, pcl, thresh_close_to_line=0.1, thresh_in_front_of_hand=0.50):

        p1, p2 = arm_xyz[0], arm_xyz[2]
        pcl_list = point_cloud2.read_points(pcl, field_names=('x', 'y', 'z'), skip_nans=True)
        valid_pts = np.array([list(k) for k in pcl_list])

        # Select points that are in the front of the hand.
        dists_plane = point_plane_distance(valid_pts, p1, p2-p1)
        thresh = thresh_in_front_of_hand + np.linalg.norm(p2-p1)
        valid_idx = dists_plane >= thresh
        valid_pts = valid_pts[valid_idx]
        dists_plane = dists_plane[valid_idx]
        if valid_pts.size == 0:
            return False, None

        # Select points that are close to the pointing direction.
        dists_3d_line = point_3d_line_distance(valid_pts, p1, p2)
        valid_idx = dists_3d_line <= thresh_close_to_line
        valid_pts = valid_pts[valid_idx]
        if valid_pts.size == 0:
            return False, None
        dists_plane = dists_plane[valid_idx]
        closest_point_idx = np.argmin(dists_plane)

        # Get hit point.
        hit_point = valid_pts[closest_point_idx]
        return True, hit_point
    
    def get_2d_hit_point(self, xyz_hit, camera_intrinsics):
        ''' Project a point represented in camera coordinate onto the image plane.
        Arguments:
            xyz_in_camera {np.ndarray}: (3, ).
            camera_intrinsics {np.ndarray}: 3x3.
        Return:
            xy {np.ndarray, np.float32}: (2, ). Column and row index.
        '''
        pt_3d_on_cam_plane = xyz_hit/xyz_hit[2]  # z=1
        xy = camera_intrinsics.dot(pt_3d_on_cam_plane)[0:2]
        xy = tuple(int(v) for v in xy)
        return xy
    
    def calculate_corners(self, center_pose, size):
        center = np.array([center_pose.position.x, center_pose.position.y, center_pose.position.z])
        half_size = np.array([size.x/2.0, size.y/2.0, size.z/2.0])
        min_corner = center - half_size
        max_corner = center + half_size
        return np.array(min_corner), np.array(max_corner)
    
    def is_ray_bbox_intersection(self, p1, p2, bbox) -> bool:
        #self.get_logger().info('comprobacion interseccion del rayo con la bbox')
        min_corner, max_corner = self.calculate_corners(bbox.center, bbox.size)
        p1_coords = np.array([p1.point.x, p1.point.y, p1.point.z])
        p2_coords = np.array([p2.point.x, p2.point.y, p2.point.z])

        direction = p2_coords - p1_coords
        direction /= np.linalg.norm(direction)
        distances = []
        for i in range (3):
            if direction[i] != 0:
                t1 = (min_corner[i] - p1_coords[i]) / direction[i]
                t2 = (max_corner[i] - p1_coords[i]) / direction[i]
                distances.extend([t1, t2])

        intersection_points = []
        for t in distances:
            intersection_point = p1_coords + t * direction
            intersection_points.append(intersection_point)


        for point in intersection_points:
            if all(min_corner[i] <= point[i] <= max_corner[i] for i in range(3)):
                return True
            
        return False
    

    def pointing_cb(self, msg_object_detections: DetectionArray, msg_detections_3d: DetectionArray) -> None:

        #self.get_logger().info('PCL, CAMERA INFO Y DETECTIONS')
        markers = MarkerArray()
        if msg_detections_3d.detections:
            human_detections = msg_detections_3d.detections

            for human_pose in human_detections:
                #self.get_logger().info('UNA PERSONA')
                pointing_msg = Pointing()

                right_arm = self.get_right_arm(human_pose.keypoints3d.data)
                #self.get_logger().info('COGE BRAZO DCHO')

                #arm exists
                if len(right_arm) == 3:
                    #self.get_logger().info('DETECTA 3 KEYPOINTS DEL BRAZO')

                    right_arm_xyz = []
                    for k in right_arm:
                        #x_coord, y_coord = k.point.x, k.point.y
                        #xp, yp, zp = point2dto3d(msg_pcl, x_coord, y_coord)
                        right_arm_xyz.append([k.point.x, k.point.y, k.point.z])



                        #marker = create_marker_point(msg_pcl.header.frame_id, xp, yp, zp)
                        #marker.header.stamp = msg_pcl.header.stamp
                        #marker.id = len(markers.markers)
                        #markers.markers.append(marker)

                    #self.get_logger().info('COMPRUEBA SI ESTA ESTIRADO')
                    is_pointing = self.is_arm_stretched(np.array(right_arm_xyz))

                    if is_pointing:
                        self.get_logger().info("IS POINTING")

                        #pointing_msg.header = msg_pcl.header
                        pointing_msg.src.id = int(human_pose.id)
                        #self.get_logger().info(f'DATA KEYPOINT {human_pose.keypoints3d.data[0].point.x}')
                        pointing_msg.src = human_pose.keypoints3d.data[0]
                        
                        #ray = create_3d_pointing_ray(msg_pcl.header.frame_id, np.array(right_arm))
                        #ray.header.stamp = msg_pcl.header.stamp
                        #ray.id = len(markers.markers)
                        #markers.markers.append(ray)
                        #is_hit, xyz_hit = self.get_3d_ray_hit_point(np.array(right_arm), msg_pcl)
                        #if is_hit:
                            #xp, yp, zp = xyz_hit
                            #marker = create_marker_point(msg_pcl.header.frame_id, float(xp), float(yp), float(zp))
                            #marker.header.stamp = msg_pcl.header.stamp
                            #marker.id = len(markers.markers)
                            #markers.markers.append(marker)

                            #pasar de 3d a 2d
                            #intrinsic_matrix = get_intrinsic_matrix(msg_camera_info)
                            #xy_hit_2d = self.get_2d_hit_point(xyz_hit, intrinsic_matrix)

                            # Comprobar si el punto esta en alguno de los objetos detectados 
                        for detection in msg_object_detections.detections:
                            #self.get_logger().info('UNA BOLSA')
                            bbox = detection.bbox3d
                            if self.is_ray_bbox_intersection(right_arm[0], right_arm[2], bbox):
                                #if is_xy_in_bbox(xy_hit_2d, bbox):
                                self.get_logger().info(f'THE BAG IS {detection.id}')
                                pointing_msg.object = bbox
                                self._pointing_pub.publish(pointing_msg)
                    
            #self._markers_pub.publish(markers)


def main():
    rclpy.init()
    node = PointingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
