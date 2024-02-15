#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Two functions:
def point_plane_distance
def point_3d_line_distance
def point2dto3d
def get_intrinsic_matrix
'''

import math
import numpy as np
import struct
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2
from yolov8_msgs.msg import BoundingBox2D, BoundingBox3D
from geometry_msgs.msg import Pose

def point_plane_distance(points, plane_point, plane_normal):
    ''' Calculate Distance between point and plane.
    Arguments:
        points {np.ndarray}: (N, 3), [[x1, y1, z1], [x2, y2, z2], ...]
        plane_point {np.ndarray}: (3, ), (x0, y0, z0)
        plane_normal {np.ndarray}: (3, ), (A, B, C)
    Return:
        dists {np.ndarray}: (N, )
            The distance might be positive or negative.
            It's positive if the point is on the side of the plane normal.
    '''
    # https://mathinsight.org/distance_point_plane
    # plane normal vector: (A, B, C)
    # plane point Q=(x0, y0, z0)
    # plane eq: A(x−x0)+B(y−y0)+C(z−z0)=0
    # plane eq: Ax+By+Cx+D=0, where D=-Ax0-By0-Cz0
    D = -plane_normal.dot(plane_point)
    numerator = points.dot(plane_normal) + D  # (N, 1)
    denominator = np.linalg.norm(plane_normal)  # (1, )
    dists = numerator / denominator
    return dists

def point_3d_line_distance(points, p1_on_3d_line, p2_on_3d_line):
    ''' Calculate the distance between 3D line and each point. 
    Arguments:
        p1_on_3d_line {np.ndarray}: shape=(3, ).
        p2_on_3d_line {np.ndarray}: shape=(3, ).
        points {np.ndarray}: shape=(N, 3).
            the points that we want to compute the distance to the 3d line.
    Return:
        dists {np.ndarray}: shape=(N, ).
            All distances are non-negative.
    '''
    # http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    crs = np.cross((points - p1_on_3d_line), (points - p2_on_3d_line))
    dists = np.linalg.norm(crs, axis=1)  # (N, )
    dists /= np.linalg.norm(p2_on_3d_line - p1_on_3d_line)
    return dists

def point2dto3d(pcl, x, y):

    is_bigendian = pcl.is_bigendian
    point_size = pcl.point_step
    row_step = pcl.row_step
            
    data_format = ""
            
    if is_bigendian:
        data_format = ">f"
    else:
        data_format = "<f"

    xp = yp = zp = 0.0

    if x >= 0 and y >= 0 and (y * row_step) + (x * point_size) + point_size <= len(pcl.data):
        xp = struct.unpack_from(data_format, pcl.data, (y * row_step) + (x * point_size))[0]
        yp = struct.unpack_from(data_format, pcl.data, (y * row_step) + (x * point_size) + 4)[0]
        zp = struct.unpack_from(data_format, pcl.data, (y * row_step) + (x * point_size) + 8)[0]

    return [xp, yp, zp]

def get_3d_bbox(pcl: PointCloud2, bbox_2d: BoundingBox2D) -> BoundingBox3D:
        
        x = int(bbox_2d.center.position.x)
        y = int(bbox_2d.center.position.y)

        x_min = y_min = z_min = float('inf')
        x_max = y_max = z_max = float('-inf')

        #center point bbox
        cp_x, cp_y, cp_z = point2dto3d(pcl, x, y)

        #bbox limits
        left_max_x, left_max_y, left_max_z = point2dto3d(pcl, int(bbox_2d.center.position.x - bbox_2d.size_x / 2.0),
                        int(bbox_2d.center.position.y + bbox_2d.size_y / 2.0))
        left_min_x, left_min_y, left_min_z = point2dto3d(pcl, int(bbox_2d.center.position.x - bbox_2d.size_x / 2.0),
                        int(bbox_2d.center.position.y - bbox_2d.size_y / 2.0))
        right_max_x, right_max_y, right_max_z = point2dto3d(pcl, int(bbox_2d.center.position.x + bbox_2d.size_x / 2.0),
                        int(bbox_2d.center.position.y + bbox_2d.size_y / 2.0))
        right_min_x, right_min_y, right_min_z = point2dto3d(pcl, int(bbox_2d.center.position.x + bbox_2d.size_x / 2.0),
                        int(bbox_2d.center.position.y - bbox_2d.size_y / 2.0))
        
        weight = (right_max_x - left_max_x) if not math.isnan((right_max_x - left_max_x)) else (right_min_x - left_min_x)
        high = (left_max_y - left_min_y) if not math.isnan((left_max_y - left_min_y)) else (right_max_y - right_min_y)

        for point in point_cloud2.read_points(pcl, field_names=("x", "y", "z"), skip_nans=True):
            dist_x = abs(point[0] - cp_x)
            dist_y = abs(point[1] - cp_y)

            #get the depth of the points near the center point
            if dist_x <=0.02 and dist_y <=0.02:
                z = float(point[2])
                z_min = min(z_min, z)
                z_max = max(z_max, z) 
        

        bbox_3d = BoundingBox3D()

        center_point = Pose()
        center_point.position.x = cp_x
        center_point.position.y = cp_y
        center_point.position.z = cp_z
        bbox_3d.center = center_point
        bbox_3d.size.x = weight
        bbox_3d.size.y = high
        bbox_3d.size.z = z_max - z_min

        return bbox_3d

def is_xy_in_bbox(xy, bbox) -> bool:
        x, y = xy[0], xy[1]
        xmin = round(bbox.center.position.x - bbox.size_x / 2.0)
        ymin = round(bbox.center.position.y - bbox.size_y / 2.0)
        xmax = round(bbox.center.position.x + bbox.size_x / 2.0)
        ymax = round(bbox.center.position.y + bbox.size_y / 2.0)
        if x >= xmin and x <= xmax and y >= ymin and y <= ymax: 
            return True
        return False

def get_intrinsic_matrix(camera_info):

    fx = camera_info.k[0]
    fy = camera_info.k[4]
    cx = camera_info.k[2]
    cy = camera_info.k[5]

    intrinsic_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])
    
    return intrinsic_matrix

def get_euclidean_distance(p1, p2):
    return math.sqrt((p2.x -p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)