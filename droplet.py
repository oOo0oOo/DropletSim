
import math, time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

def intersect_lines(p1, p2, p3, p4, check_second = True):
	u = ((p1[0] - p2[0])*(p3[1] - p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0]))
	
	if u > 0:
		x = (p1[0]*p2[1] - p1[1]*p2[0])*(p3[0] - p4[0]) - (p1[0] - p2[0])*(p3[0]*p4[1] - p3[1]*p4[0])
		x /= u
		y = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[1] - p4[1])) - ((p1[1] - p2[1])*(p3[0]*p4[1] - p3[1]*p4[0]))
		y /= u

		# Check if on second line segment or outside
		# Not very versatile...
		if check_second:
			# give it some elasticity
			f = 0.99
			if min(p3[0], p4[0]) * f <= x <= max(p3[0], p4[0]) * (2-f) and min(p3[1], p4[1]) * f <= y <= max(p3[1], p4[1]) * (2-f):
				return [x, y]
		else:
			return [x, y]

	return False

class DropletAnimation(object):
	def __init__(self, barriers, start_point = (50, 30), num_spikes = 50, area = 7000, max_dist = 100):
		self.center_point = start_point
		self.barriers = barriers
		self.num_spikes = num_spikes
		self.area = area
		self.max_len_spike = max_dist

		self._spikes = np.array([0.1 for i in xrange(self.num_spikes)])
		self._angle = math.radians(360. / self.num_spikes)
		self._mult_term = math.sin(self._angle) * 0.5
		self._max_dist = np.copy(self._spikes)

	def get_diameter(self):
		return 2 * sum(self._spikes)/self.num_spikes

	def get_area(self):
		return np.sum(self._spikes * np.roll(self._spikes, 1) * self._mult_term)

	def get_area_pure(self):
		area = 0
		for i, p in enumerate(self._spikes):
			area += p * self._spikes[(i+1)%self.num_spikes] * self._mult_term
		return area

	def move_up_spikes(self, distance):
		to_move = self._spikes < self._max_dist
		self._spikes[to_move] += distance
		#at_max = np.logical_not(to_move)
		#self._spikes[at_max] = self._max_dist[at_max]

	def move_center_point(self, t):
		cp = self.center_point
		self.center_point = (cp[0] + t[0], cp[1] + t[1])
		self.reset_max_dist()
		self.find_shape()

	def get_shape(self):
		x, y = [], []
		for i, p in enumerate(self._spikes):
			x.append((p * math.cos(i*self._angle)) + self.center_point[0])
			y.append((p * math.sin(i*self._angle)) + self.center_point[1])
		return x, y

	def get_max(self):
		x, y = [], []
		for i, p in enumerate(self._max_dist):
			x.append((p * math.cos(i*self._angle)) + self.center_point[0])
			y.append((p * math.sin(i*self._angle)) + self.center_point[1])
		return x, y

	def get_max_area(self):
		return np.sum(self._max_dist * np.roll(self._max_dist, 1) * self._mult_term)

	def find_shape(self, point = False):
		self._spikes = np.array([0.1 for i in xrange(self.num_spikes)])

		if point and type(point) == tuple and len(point) == 2:
			self.center_point = point

		if self.area < self.get_max_area():
			v = d.get_area()
			n  = 0
			while v < self.area:
				n += 1
				d.move_up_spikes(1)
				v = d.get_area()
			return n
		else:
			print 'Area too large!!'
			self._spikes = np.copy(self._max_dist)

	def reset_max_dist(self):
		cp = self.center_point

		for i in xrange(self.num_spikes):
			x = (self.max_len_spike * math.cos(i*self._angle)) + cp[0]
			y = (self.max_len_spike * math.sin(i*self._angle)) + cp[1]
			outer = (x, y)

			len_inter = self.max_len_spike
			inter = False

			for p1, p2 in self.barriers:
				if ((p2[0] - p1[0])*(cp[1] - p1[1]) - (p2[1] - p1[1])*(cp[0] - p1[0])) > 0:
					p_inter = intersect_lines(cp, outer, p1, p2)
				else:
					p_inter = intersect_lines(outer, cp, p1, p2)

				if p_inter:
					# calculate distance to intersection point
					vect = [p_inter[0] - cp[0], p_inter[1] - cp[1]]
					dist = math.sqrt((vect[0]**2) + (vect[1]**2))
					if dist < len_inter:
						len_inter = dist
						inter = p_inter

			if inter:
				self._max_dist[i] = len_inter
			else:
				self._max_dist[i] = self.max_len_spike

def update_plot(i, d, droplet):	
	d.move_center_point((1, 0))
	d.find_shape(7000)
	x, y = d.get_shape()
	offsets = [[x[i], y[i]] for i in xrange(len(x))]
	droplet.set_offsets(offsets)
	'''
		x, y = d.get_max()
		offsets = [[x[i], y[i]] for i in xrange(len(x))]
		limits.set_offsets(offsets)
	'''

if __name__ == '__main__':
	# Setup environment
	barriers = [
		# The entry channel
		[[-100, 100], [100, 100]],
		[[-100, 50], [100, 50]],

		# The expansion
		[[100, 100], [200, 150]],
		[[100, 50], [200, 0]],

		# The constriction
		[[200, 150], [300, 90]],
		[[200, 0], [300, 50]],

		# The exit channel
		[[300, 90], [600, 90]],
		[[300, 50], [600, 40]]
	]

	start_point = (50, 75)

	# Calc first frame
	d = DropletAnimation(barriers, start_point, num_spikes = 50, max_dist = 150,
			area = 7000)
	d.move_center_point((0,0))

	fig = plt.figure()

	# Plot all the barriers
	for p1, p2 in barriers:
		plt.plot([p1[0], p2[0]], [p1[1], p2[1]], lw = 2, c='black')

	#backgrounds = fig.canvas.copy_from_bbox(ax.bbox)

	x, y = d.get_shape()
	droplet = plt.scatter(x, y, s=10, c='red')

	'''
		x, y = d.get_max()
		limits = plt.scatter(x, y, s=10, c='blue')
	'''

	ani = animation.FuncAnimation(fig, update_plot, fargs=(d, droplet))
	plt.axis('equal')
	plt.show()