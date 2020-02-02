import os
import csv
import numpy as np

import torch.utils.data as data

class MOTDataset(data.Dataset):
    def __init__(self, root_dir="/home/deep/enis/cv3dst_exercise/data/MOT16/train/", seq_name="MOT16-02"):
        self.root = root_dir + seq_name

        self.flows = np.load(os.path.join(self.root, 'flow.npy'))
        gt_file = os.path.join(self.root, 'gt', 'gt.txt')

        self.im_boxes = {}
        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                im_index = int(row[0])
                if im_index not in self.im_boxes:
                    self.im_boxes[im_index] = {}
                track_id = int(row[1])
                self.im_boxes[im_index][track_id] = [
                    int(row[2]) - 1,
                    int(row[3]) - 1 - 28,
                    int(row[2]) + int(row[4]) - 1,
                    int(row[3]) + int(row[5]) - 1 - 28,
                ]


    def __getitem__(self, idx):
        flow = self.flows[idx]
        # flow = np.concatenate((flow, np.zeros((1, flow.shape[1], flow.shape[2]))), axis=0) 
        prev_img = idx + 1
        curr_img = idx + 2

        prev_tracks = self.im_boxes[prev_img]
        curr_tracks = self.im_boxes[curr_img]

        prev_set = set(list(prev_tracks.keys()))
        curr_set = set(list(curr_tracks.keys()))

        prev_boxes = []
        curr_boxes = []

        for track_id in prev_set.intersection(curr_set):
            prev_boxes.append(prev_tracks[track_id])
            curr_boxes.append(curr_tracks[track_id])

        prev_boxes = np.array(prev_boxes)
        curr_boxes = np.array(curr_boxes)

        return [flow.astype(np.float32), prev_boxes.astype(np.float32)], curr_boxes.astype(np.float32)

    def __len__(self):
        return len(self.flows)