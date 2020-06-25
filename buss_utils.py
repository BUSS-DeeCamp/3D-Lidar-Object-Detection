import os
import json
import numpy as np
from typing import List
from scipy.spatial import Delaunay

def get_label_annos(content: List[dict]) -> dict:
    annotations = {
        'location': [],
        'dimension': [],
        'rotation': [],
        'class_name': [],
        'class_id': [],
        'num_points': [],
        
    }
    annotations['location'] = np.array(
        [x['location'] for x in content]).reshape(-1, 3)
    annotations['dimension'] = np.array(
        [x['dimension'] for x in content]).reshape(-1, 3)
    annotations['rotation'] = np.array(
        [x['rotation'] for x in content]).reshape(-1, 3)
    annotations['class_name'] = np.array(
        [x['class_name'] for x in content])
    annotations['class_id'] = np.array(
        [x['class_id'] for x in content])
    annotations['num_points'] = np.array(
        [x['num_points'] for x in content])
    
   
    return annotations

def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['class_name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info

















def boxes3d_to_corners3d_lidar(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords, see the definition of ry in KITTI dataset
    :param z_bottom: whether z is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    w, l, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    y_corners = np.array([-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.], dtype=np.float32).T
    if bottom_center:
        z_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        z_corners[:, 4:8] = h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        z_corners = np.array([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), -np.sin(ry), zeros],
                         [np.sin(ry), np.cos(ry),  zeros],
                         [zeros,      zeros,        ones]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)

def enlarge_box3d(boxes3d, extra_width):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    large_boxes3d[:, 2] -= extra_width  # bugfixed: here should be minus, not add in LiDAR, 20190508
    return large_boxes3d


def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag


def rotate_pc_along_z(pc, rot_angle):
    """
    params pc: (N, 3+C), (N, 3) is in the LiDAR coordinate
    params rot_angle: rad scalar
    Output pc: updated pc with XYZ rotated
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, 0:2] = np.dot(pc[:, 0:2], rotmat)
    return pc

def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    points = points[mask]
    return points

def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds