#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from yolov8_msgs.msg import BoundingBox3D

def create_3d_pointing_ray(frame, arm_xyz, extend_meters_3d=3.0) -> Marker:
    p1, p2 = arm_xyz[0], arm_xyz[2]
    direction_vector = np.array(p2) - np.array(p1)
    p3 = p1 + extend_meters_3d * direction_vector / np.linalg.norm(direction_vector)
    return create_marker_line(frame, p1[0], p1[1], p1[2], p3[0], p3[1], p3[2])

def create_marker_point(frame, x, y, z) -> Marker:

    marker = Marker()
    marker.header.frame_id = frame
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.frame_locked = True

    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05

    marker.color.b = 1.0
    marker.color.a = 1.0

    marker.lifetime = Duration(seconds=0.5).to_msg()

    return marker

def create_marker_line(frame, x1, y1, z1, x2, y2, z2) -> Marker:

    marker_line = Marker()
    marker_line.header.frame_id = frame
    marker_line.type = Marker.LINE_STRIP
    marker_line.action = Marker.ADD
    marker_line.frame_locked = True
    
    marker_line.color.a = 1.0
    marker_line.scale.x = 0.05
    marker_line.color.r = 1.0
    
    marker_line.lifetime = Duration(seconds=0.1).to_msg()
    
    start_point = Point()
    end_point = Point()
    
    start_point.x = x1
    start_point.y = y1
    start_point.z = z1
    
    end_point.x = x2 
    end_point.y = y2 
    end_point.z = z2 
    
    marker_line.points.append(start_point)
    marker_line.points.append(end_point)
    
    return marker_line

def create_marker_cube(frame, bbox_3d: BoundingBox3D) -> Marker:
    
    marker = Marker()
    marker.header.frame_id = frame
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    marker.pose.position.x = bbox_3d.center.position.x
    marker.pose.position.y = bbox_3d.center.position.y
    marker.pose.position.z = bbox_3d.center.position.z

    marker.scale.x = bbox_3d.size.x
    marker.scale.y = bbox_3d.size.y
    marker.scale.z = bbox_3d.size.z

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.5

    marker.lifetime = Duration(seconds=1.0).to_msg()

    return marker