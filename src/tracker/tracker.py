import torch
import torch.nn.functional as F

import numpy as np
import motmetrics as mm
from torchvision.ops.boxes import clip_boxes_to_image, nms

import matplotlib.pyplot as plt


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, self.w-self.tw:]

class Tracker:
	"""The main tracking file, here is where magic happens."""

	def __init__(self, obj_detect, use_linear=True, flownet=None):
		self.obj_detect = obj_detect
		self.flownet = flownet
		self.use_linear = use_linear

		self.tracks = []
		self.track_actives = []
		self.track_inactives = []

		self.track_num = 0
		self.im_index = 0
		self.results = {}

		self.img_prev = None
		self.img = None

		self.mot_accum = None

	def reset(self, hard=True):
		self.track_actives = []
		self.track_inactives = []
		self.tracks = []
		self.img_prev = None
		self.img = None

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def add(self, new_boxes, new_scores):
		"""Initializes new Track objects and saves them."""
		num_new = len(new_boxes)
		for i in range(num_new):
			self.track_actives.append(Track(
			# self.tracks.append(Track(
				new_boxes[i].cpu().numpy(),
				new_scores[i].cpu().numpy(),
				self.track_num + i
			))
		self.track_num += num_new

	
	def get_linear_box(self, track):
		if len(track.prev_boxes) > 3:
			x1s = [b[0] for b in track.prev_boxes[-3:]]
			x2s = [b[2] for b in track.prev_boxes[-3:]]
			y1s = [b[1] for b in track.prev_boxes[-3:]]
			y2s = [b[3] for b in track.prev_boxes[-3:]]

			def get_point(points):
				A = np.vstack([[0, 1, 2], np.ones(3)]).T
				m, b = np.linalg.lstsq(A, points, rcond=None)[0]
				return m * 3 + b

			x1 = get_point(x1s)
			x2 = get_point(x2s)
			y1 = get_point(y1s)
			y2 = get_point(y2s)

			return torch.Tensor([x1, y1, x2, y2])
			
		return torch.Tensor(track.box)

	def get_linear_boxes(self):
		box = []
		for t in self.track_actives:
			box.append(self.get_linear_box(t))

		box = torch.stack([torch.Tensor(t) for t in box], 0)
		box = clip_boxes_to_image(box, self.img.shape[-2:])
		return box

	def divide_box_flow(self, box_flow, size):
		w, h = size
		# top_left = np.mean(box_flow[:, :h//2, :w//2], axis=(1, 2))
		# bottom_left = np.mean(box_flow[:, h//2:, :w//2], axis=(1, 2))
		# top_right = np.mean(box_flow[:, :h//2, w//2:], axis=(1, 2))
		# bottom_right = np.mean(box_flow[:, h//2:, w//2:], axis=(1, 2))

		# x0_change = min(top_left[0] / 1000, bottom_left[0] / 1000)
		# y0_change = min(top_left[1] / 1000, top_right[1] / 1000)
		# x1_change = max(bottom_right[0] / 1000, top_right[0] / 1000)
		# y1_change = max(bottom_right[1] / 1000, bottom_left[1] / 1000)

		# top = np.mean(box_flow[:, :h//2, :], axis=(1, 2))
		# bottom = np.mean(box_flow[:, h//2:, :], axis=(1, 2))
		# right = np.mean(box_flow[:, :, w//2:], axis=(1, 2))
		# left = np.mean(box_flow[:, :, :w//2], axis=(1, 2))

		# x0_change = left[0] / 1000
		# y0_change = top[1] / 1000
		# x1_change = right[0] / 1000
		# y1_change = bottom[1] / 1000

		# top_left = np.mean(box_flow[:, :h//2, :w//2], axis=(1, 2))
		# bottom_right = np.mean(box_flow[:, h//2:, w//2:], axis=(1, 2))

		# x0_change = top_left[1] / 1000
		# y0_change = top_left[0] / 1000
		# x1_change = bottom_right[1] / 1000
		# y1_change = bottom_right[0] / 1000

		# mean_x, mean_y = np.mean(box_flow[:, :, :], axis=(1, 2))

		# x0_change = mean_x / 1000
		# y0_change = mean_y / 1000
		# x1_change = mean_x / 1000
		# y1_change = mean_y / 1000
		
		centre_flow = box_flow[:, h//2 - h//8:h//2 + h//8, w//2 - w//8:w//2 + w//8]

		mean_x, mean_y = np.mean(centre_flow, axis=(1, 2))

		x0_change = mean_x / 2000
		y0_change = mean_y / 2000
		x1_change = mean_x / 2000
		y1_change = mean_y / 2000

		return np.array([x0_change, y0_change, x1_change, y1_change])


	def get_flownet_boxes(self):
		box = []

		images = [self.img.numpy()[0].transpose(1, 2, 0), self.img_prev.numpy()[0].transpose(1, 2, 0)]
		image_size = self.img.numpy()[0].transpose(1, 2, 0).shape[:2]
		cropper = StaticCenterCrop(image_size, [1024, 1920])
		images = list(map(cropper, images))

		images = np.array(images).transpose(3,0,1,2)
		images = torch.from_numpy(images.astype(np.float32))
		images = images.unsqueeze(0).cuda()

		flow = self.flownet(images)[0].cpu().detach().numpy()

		for t in self.track_actives:
			t_box = t.box
			x1, y1, x2, y2 = np.array(t_box).astype(int) - np.array([0, 28, 0, 28])
			size = (x2 - x1, y2 - y1)

			if size[0] * size[1] < 16000:
				box.append(self.get_linear_box(t))
				continue

			box_flow = flow[:, y1:y2, x1:x2]
			diff = self.divide_box_flow(box_flow, size)

			# print("#", t_box, "%", (t_box + diff))
			box.append(torch.Tensor(t.box + diff))
		box = torch.stack([torch.Tensor(t) for t in box], 0)
		box = clip_boxes_to_image(box, self.img.shape[-2:])
		return box

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.track_actives) == 1:
			box = torch.Tensor(self.track_actives[0].box).unsqueeze(0)
		elif len(self.track_actives) > 1:
			if self.flownet is not None and self.img_prev is not None:
				return self.get_flownet_boxes()
			if self.use_linear:
				return self.get_linear_boxes()

			box = torch.stack([torch.Tensor(t.box) for t in self.track_actives], 0)
		else:
			box = torch.zeros(0).cuda()
		return box

	def get_inactive_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.track_inactives) == 1:
			box = torch.Tensor(self.track_inactives[0].box).unsqueeze(0)
		elif len(self.track_inactives) > 1:
			box = torch.stack([torch.Tensor(t.box) for t in self.track_inactives], 0)
		else:
			box = torch.zeros(0).cuda()
		return box

	def data_association(self, boxes, scores):
		self.tracks = []
		self.add(boxes, scores)

	def ignore_small_box(self, box):
		return ((box[2] - box[0]) * (box[3] - box[1])) < 625
		
	def step(self, frame):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		# print(self.track_actives)
		# print(self.track_inactives)
		# print('------')
		# # Inactive Part
		# boxes, scores = self.obj_detect.detect_with_proposal(frame['img'], self.get_inactive_pos())
		# boxes = clip_boxes_to_image(boxes, frame['img'].shape[-2:])

		# results = list(zip(boxes.cpu().numpy(), scores.cpu().numpy()))

		# B = []
		# S = []
		# may_active = []

		# for i, t_inactive in enumerate(self.track_inactives):
		# 	b_t, s_t = results[i]

		# 	t_inactive.inactive_count +=1

		# 	if s_t > 0.5:
		# 		may_active.append(t_inactive)
		# 		B.append(b_t)
		# 		S.append(s_t)
		# 		continue

		# B = np.array(B)
		# S = np.array(S)
		# kept_indices = nms(torch.Tensor(B), torch.Tensor(S), 0.6) # lambda_active

		# may_active = [may_active[i] for i in kept_indices] # list(np.array(may_active)[kept_indices])

		# B = np.array(B)[kept_indices] if len(kept_indices) else B

		# if B.ndim != 0 and B.ndim != 2:
		# 	B = np.expand_dims(B, axis=0)

		# for t_active, b_t in zip(may_active, B):
		# 	t_active.box = b_t
		# 	self.track_actives.append(t_active)

		# may_inds = [t.id for t in may_active]
		# self.track_inactives = [t for t in self.track_inactives if t.id not in may_inds or t.inactive_count < 3]


		self.img = frame['img']
		# Track actives
		B = []
		S = []
		results = []
		boxes, scores = self.obj_detect.detect_with_proposal(frame['img'], self.get_pos())
		boxes = clip_boxes_to_image(boxes, frame['img'].shape[-2:])

		results = list(zip(boxes.cpu().numpy(), scores.cpu().numpy()))

		removed_actives = []
		for i, t_active in enumerate(self.track_actives):
			b_t, s_t = results[i]

			if s_t > 0.5:
				B.append(b_t)
				S.append(s_t)
				continue
			
			self.track_inactives.append(t_active)
			removed_actives.append(i)
		self.track_actives = [t for i, t in enumerate(self.track_actives) if i not in removed_actives]

		B = np.array(B)
		S = np.array(S)
		kept_indices = nms(torch.Tensor(B), torch.Tensor(S), 0.6) # lambda_active

		for i, t_active in enumerate(self.track_actives):
			if i not in kept_indices:
				self.track_inactives.append(t_active)

		self.track_actives = [self.track_actives[i] for i in kept_indices]

		B = [B[i] for i in kept_indices]

		for t_active, b_t in zip(self.track_actives, B):
			t_active.prev_boxes.append(b_t)
			t_active.box = b_t

		# Add new objects
		boxes, scores = self.obj_detect.detect(frame['img'])

		inds = torch.nonzero(scores > 0.5).squeeze(1)
		boxes, scores = boxes[inds], scores[inds]

		# Too Slow
		# removed_boxes = []
		# for i, box in enumerate(boxes):
		# 	for b_t in [t.box for t in self.track_actives]:
		# 		iou_score = bb_intersection_over_union(box, b_t)
		# 		if iou_score > 0.4: # lambda_new
		# 			removed_boxes.append(i)
		# boxes = [b for i, b in enumerate(boxes) if i not in removed_boxes]
		# scores = [s for i, s in enumerate(scores) if i not in removed_boxes]

		if len(self.track_actives):
			track_boxes = np.stack([t.box for t in self.track_actives], 0)
			track_boxes[:, 2] = track_boxes[:, 2] - track_boxes[:, 0] # width
			track_boxes[:, 3] = track_boxes[:, 3] - track_boxes[:, 1] # heigth

			boxes_wh = boxes.clone()
			boxes_wh[:, 2] = boxes_wh[:, 2] - boxes_wh[:, 0] # width
			boxes_wh[:, 3] = boxes_wh[:, 3] - boxes_wh[:, 1] # heigth

			distances = mm.distances.iou_matrix(boxes_wh, track_boxes, 1 - 0.4)
			inds = np.all(np.isnan(distances), axis=1)
			boxes = boxes[inds]
			scores = scores[inds]


		inds = [i for i in range(len(boxes)) if self.ignore_small_box(boxes[i])]

		boxes = [b for i, b in enumerate(boxes) if i not in inds]
		scores = [s for i, s in enumerate(scores) if i not in inds]

		self.add(boxes, scores)

		for t in self.track_actives:
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			self.results[t.id][self.im_index] = np.concatenate([t.box, np.array([t.score])])

		self.im_index += 1
		self.img_prev = frame['img']

	def get_results(self):
		return self.results


class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, box, score, track_id):
		self.id = track_id
		self.box = box
		self.prev_boxes = [box]
		self.score = score
		self.inactive_count = 0

	def __str__(self):
		print(self.id, self.box)
