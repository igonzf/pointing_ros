import math
import numpy as np


def calculate_bbox_v(bbox):
    center = [bbox.center.position.x, bbox.center.position.y, bbox.center.position.z]
    dimensions = [bbox.size.x, bbox.size.y, bbox.size.z]

    m_w, m_h, m_l = np.array(dimensions) / 2

    offsets = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ])
    
    v_list = offsets * np.array([m_w, m_l, m_h])

    bbox_v = v_list + center

    return bbox_v

def calculate_bbox_faces(bbox_v):
    bbox_faces = []
    index = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [0,3,7,4], [1,2,6,5]]

    for i in index:
            face = [bbox_v[j] for j in i]
            bbox_faces.append(face)

    return bbox_faces

def calculate_plane_eq (p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    a, b, c = normal
    d = -np.dot(normal, p1)
    return a, b, c, d

def calculate_line_plane_intersection(p1, p2, plane_eq):
    a, b, c, d = plane_eq
    u = p2 - p1
    t = (-d - np.dot(np.array([a, b, c], p1)) / np.dot(np.array([a, b, c]), u))
    intersection_point = p1 + t * u

    return intersection_point

def is_point_inside_bbox_face(p, face) -> bool:
    p0, p1, p2, p3 = face
    v0 = p1 - p0
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p0 - p3

    n0 = np.cross(v0, p - p0)
    n1 = np.cross(v1, p - p1)
    n2 = np.cross(v2, p - p2)
    n3 = np.cross(v3, p - p3)

    if(np.dot(n0, n1)>=0 and np.dot(n1, n2)>=0 and np.dot(n2, n3)>=0 and np.dot(n3, n0)>=0):
        return True
    
    return False

def is_intersection(p1, p3, bbox_faces):
    direction = p3 - p1

    for face in bbox_faces:
        plane_eq = calculate_plane_eq(*face[:3])
        normal_plane = np.array(plane_eq[:3])

        if np.abs(np.dot(direction, normal_plane)) < 1e-6:
            continue
        
        intersection_point = calculate_line_plane_intersection(p1, p3, plane_eq)
        if is_point_inside_bbox_face(intersection_point, face):
            return True
    return False