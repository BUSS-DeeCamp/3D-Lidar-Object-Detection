import os
import json
import numpy as np
from typing import List
import buss_utils
import copy
import spconv

class BussDataset():
    def __init__(self, root_path, class_names, split, training = True):
        """
        :param root_path: DeeCamp data path
        :param split:
        """
        self.index_of_classes = ['Car', 'Truck', 'Tricar', 'Cyclist', 'Pedestrain']
        self.root_path = root_path
        self.split = split

        if split in ['train', 'val', 'test']:
            self.split_dir = os.path.join(self.root_path, 'labels_filer/', split + '_filter.txt')
            #print(split_dir)
        self.class_names = class_names
        self.training = training
        #self.logger = logger

        self.mode = 'TRAIN' if self.training else 'TEST'

        self.buss_infos = []
        #self.include_kitti_data(self.mode, logger)
        # self.kitti_infos = self.kitti_infos[:100]
        self.dataset_init()
        #print(self.__len__())
        self.__getitem__(352)
        
    
    def dataset_init(self):
        with open(self.split_dir) as rf:
            for line in rf:
                data_dict = json.loads(line.strip())
                self.buss_infos.append(data_dict)

        #print(self.buss_infos[0])

    def get_lidar(self, ipath):
        lidar_file = os.path.join(self.root_path, ipath)
        assert os.path.exists(lidar_file)
        pc = np.fromfile(lidar_file, dtype = np.float32)
        pc = pc.reshape(-1,4)
        return pc
        #return np.with open('labels_filer/val_filter.txt') as rf:
    
    def generate_voxel_part_targets(self, voxel_centers, gt_boxes, gt_classes, generate_bbox_reg_labels=True):
        """
        :param voxel_centers: (N, 3) [x, y, z]
        :param gt_boxes: (M, 7) [x, y, z, w, l, h, ry] in LiDAR coords
        :return:
        """
        

        MEAN_SIZE = {
                    'Car': [1.6, 3.9, 1.56],
                    'Pedestrian': [0.6, 0.8, 1.73],
                    'Cyclist': [0.6, 1.76, 1.73],
                    'Tricar' : [1.4, 3.1, 1.56],
                    'Truck' : [1.8, 4.2, 1.78]
                }

        GT_EXTEND_WIDTH = 0.2

        extend_gt_boxes = buss_utils.enlarge_box3d(gt_boxes, extra_width=GT_EXTEND_WIDTH)
        gt_corners = buss_utils.boxes3d_to_corners3d_lidar(gt_boxes)
        extend_gt_corners = buss_utils.boxes3d_to_corners3d_lidar(extend_gt_boxes)

        cls_labels = np.zeros(voxel_centers.shape[0], dtype=np.int32)
        reg_labels = np.zeros((voxel_centers.shape[0], 3), dtype=np.float32)
        bbox_reg_labels = np.zeros((voxel_centers.shape[0], 7), dtype=np.float32) if generate_bbox_reg_labels else None

        for k in range(gt_boxes.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = buss_utils.in_hull(voxel_centers, box_corners)
            fg_voxels = voxel_centers[fg_pt_flag]
            cls_labels[fg_pt_flag] = gt_classes[k]

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = buss_utils.in_hull(voxel_centers, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_labels[ignore_flag] = -1

            # part offset labels
            transformed_voxels = fg_voxels - gt_boxes[k, 0:3]
            transformed_voxels = buss_utils.rotate_pc_along_z(transformed_voxels, -gt_boxes[k, 6])
            reg_labels[fg_pt_flag] = (transformed_voxels / gt_boxes[k, 3:6]) + np.array([0.5, 0.5, 0], dtype=np.float32)

            if generate_bbox_reg_labels:
                # rpn bbox regression target
                center3d = gt_boxes[k, 0:3].copy()
                center3d[2] += gt_boxes[k][5] / 2  # shift to center of 3D boxes
                bbox_reg_labels[fg_pt_flag, 0:3] = center3d - fg_voxels
                bbox_reg_labels[fg_pt_flag, 6] = gt_boxes[k, 6]  # dy

                cur_mean_size = MEAN_SIZE[self.index_of_classes[gt_classes[k]]]
                bbox_reg_labels[fg_pt_flag, 3:6] = (gt_boxes[k, 3:6] - np.array(cur_mean_size)) / cur_mean_size

        reg_labels = np.maximum(reg_labels, 0)
        return cls_labels, reg_labels, bbox_reg_labels
    
    
    
    def prepare_data(self, input_dict, has_label=True, use_point_features=3):
        """
        :param input_dict:
            sample_idx: string
            calib: object, calibration related
            points: (N, 3 + C1)
            gt_boxes_lidar: optional, (N, 7) [x, y, z, w, l, h, rz] in LiDAR coordinate, z is the bottom center
            gt_names: optional, (N), string
        :param has_label: bool
        :return:
            voxels: (N, max_points_of_each_voxel, 3 + C2), float
            num_points: (N), int
            coordinates: (N, 3), [idx_z, idx_y, idx_x]
            num_voxels: (N)
            voxel_centers: (N, 3)
            calib: object
            gt_boxes: (N, 8), [x, y, z, w, l, h, rz, gt_classes] in LiDAR coordinate, z is the bottom center
            points: (M, 3 + C)
        """
        sample_idx = input_dict['sample_idx']
        points = input_dict['points']

        if has_label:
            gt_boxes = input_dict['gt_boxes'].copy()
            gt_names = input_dict['gt_names'].copy()
            gt_classes = input_dict['gt_namesid'].copy()
            gt_num_points = input_dict['gt_num_points'].copy()
        

        points = points[:, :use_point_features]
        if self.split == 'train' or self.split == 'val':
            np.random.shuffle(points)

        voxel_generator = spconv.utils.VoxelGenerator(
            voxel_size=[0.1, 0.1, 0.1], 
            point_cloud_range=[0, -40, -3, 70.4, 40, 1],
            max_num_points=30,
            max_voxels=40000
            )
        voxel_grid = voxel_generator.generate(points)

        # Support spconv 1.0 and 1.1
        try:
            voxels, coordinates, num_points = voxel_grid
        except:
            voxels = voxel_grid["voxels"]
            coordinates = voxel_grid["coordinates"]
            num_points = voxel_grid["num_points_per_voxel"]

        voxel_centers = (coordinates[:, ::-1] + 0.5) * voxel_generator.voxel_size \
                        + voxel_generator.point_cloud_range[0:3]

        #pick the points in a districted range
        points = buss_utils.mask_points_by_range(points, [0, -40, -3, 70.4, 40, 1])

        example = {}
        if has_label:
            if self.split == 'val':
                # for eval_utils
                selected = buss_utils.keep_arrays_by_name(gt_names, self.class_names)
                gt_boxes = gt_boxes[selected]
                gt_names = gt_names[selected]
                gt_classes = gt_classes[selected]
                gt_num_points = gt_num_points[selected]
            

            seg_labels, part_labels, bbox_reg_labels = \
                    self.generate_voxel_part_targets(voxel_centers, gt_boxes, gt_classes)
            example['seg_labels'] = seg_labels
            example['part_labels'] = part_labels
            example['bbox_reg_labels'] = bbox_reg_labels

            gt_boxes = np.concatenate((gt_boxes, gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)

            example.update({
                'gt_boxes': gt_boxes, 
                'gt_num_points': gt_num_points
            })

        example.update({
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coordinates,
            'voxel_centers': voxel_centers,
            'points': points
        })

        return example
    
    def __len__(self):
        return len(self.buss_infos)

    def __getitem__(self, index):
        # index = 4
        info = copy.deepcopy(self.buss_infos[index])

        sample_idx = info['id']
        sample_path = info['path']
        sample_md5 = info['md5']
        sample_gts = info['gts']


        points = self.get_lidar(sample_path)
        input_dict = {
            'sample_idx': sample_idx,
            'points': points,
        }
        if self.split == 'train' or self.split == 'val':
            annos = buss_utils.get_label_annos(sample_gts)
            ##type(annos) = dict
            annos = buss_utils.drop_info_with_name(info = annos, name = 'DontCare')
            loc, dims, rots = annos['location'], annos['dimension'], annos['rotation']
            gt_boxes = np.concatenate([loc, dims, rots], axis=1).astype(np.float32)
            input_dict.update({
                'gt_boxes': gt_boxes,
                'gt_names': annos['class_name'],
                'gt_namesid': annos['class_id'],
                'gt_num_points': annos['num_points']
            })


        example = self.prepare_data(input_dict=input_dict, has_label=len(input_dict)>2)

        example['sample_idx'] = sample_idx
        example['num_points'] = input_dict['gt_num_points']
        print(example)
        return example



    
if "__main__" == __name__:
    root_path = '/home/laptop2/Deecamp/dataset/testing/'
    class_names = 'Car'
    split = 'val'
    #A = BaseBussDataset(root_path, split)
    B = BussDataset(root_path, class_names, split)
