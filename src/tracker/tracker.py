import torch
import torch.nn.functional as F

import numpy as np
import motmetrics as mm
from torchvision.ops.boxes import clip_boxes_to_image, nms

def bb_intersection_over_union(boxA, boxB):
	# Source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = (interArea + 1e-8) / float(boxAArea + boxBArea - interArea + 1e-8)
 
	# return the intersection over union value
	return iou


class Tracker:
	"""The main tracking file, here is where magic happens."""

	def __init__(self, obj_detect):
		self.obj_detect = obj_detect

		self.tracks = []
		self.track_actives = []
		self.track_inactives = []

		self.track_num = 0
		self.im_index = 0
		self.results = {}

		self.mot_accum = None

	def reset(self, hard=True):
		self.track_actives = []
		self.track_inactives = []
		self.tracks = []

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

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.track_actives) == 1:
			box = torch.Tensor(self.track_actives[0].box).unsqueeze(0)
		elif len(self.track_actives) > 1:
			boxes = []
			for t in self.track_actives:
				if len(t.prev_boxes) > 3:
					x1s = [b[0] for b in t.prev_boxes[-3:]]
					x2s = [b[2] for b in t.prev_boxes[-3:]]
					y1s = [b[1] for b in t.prev_boxes[-3:]]
					y2s = [b[3] for b in t.prev_boxes[-3:]]

					def get_point(points):
						A = np.vstack([[0, 1, 2], np.ones(3)]).T
						m, b = np.linalg.lstsq(A, points, rcond=None)[0]
						return m * 3 + b

					x1 = get_point(x1s)
					x2 = get_point(x2s)
					y1 = get_point(y1s)
					y2 = get_point(y2s)

					boxes.append(torch.Tensor([x1, y1, x2, y2]))
					
				else:
					boxes.append(torch.Tensor(t.box))
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
		removed_boxes = []
		for i, box in enumerate(boxes):
			if scores[i] < 0.5: 
				removed_boxes.append(i)
				continue
			for b_t in [t.box for t in self.track_actives]:
				iou_score = bb_intersection_over_union(box, b_t)
				if iou_score > 0.4: # lambda_new
					removed_boxes.append(i)
		boxes = [b for i, b in enumerate(boxes) if i not in removed_boxes]
		scores = [s for i, s in enumerate(scores) if i not in removed_boxes]

		inds = [i for i in range(len(boxes)) if self.ignore_small_box(boxes[i])]

		boxes = [b for i, b in enumerate(boxes) if i not in inds]
		scores = [s for i, s in enumerate(scores) if i not in inds]

		self.add(boxes, scores)

		for t in self.track_actives:
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			self.results[t.id][self.im_index] = np.concatenate([t.box, np.array([t.score])])

		self.im_index += 1

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
