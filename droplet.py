import numpy as np
import math
from pylab import scatter, axis, show, plot


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
			if min(p3[0], p4[0]) <= x <= max(p3[0], p4[0]) and min(p3[1], p4[1]) <= y <= max(p3[1], p4[1]):
				return [x, y]
		else:
			return [x, y]

	return False

class DropletAnimation(object):
	def __init__(self, barriers = [], start_point = (50, 30), num_spikes = 50):
		self.center_point = start_point
		self.barriers = np.array(barriers)
		self.num_spikes = num_spikes

		self._spikes = np.array([0.1 for i in xrange(self.num_spikes)])
		self._angle = 360. / self.num_spikes
		self._spike_indices = xrange(self.num_spikes)
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

	def find_shape(self, area):
		if area < self.get_max_area():
			v = d.get_area()
			n  = 0
			while v < area:
				n += 1
				d.move_up_spikes(0.1)
				v = d.get_area()
			return n
		else:
			print 'Area too large!!'
			self._spikes = np.copy(self._max_dist)

	def set_max_dist(self, max_dist):
		cp = self.center_point

		for i in xrange(self.num_spikes):
			x = (max_dist * math.cos(i*self._angle)) + cp[0]
			y = (max_dist * math.sin(i*self._angle)) + cp[1]
			outer = (x, y)

			len_inter = max_dist
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
				self._max_dist[i] = max_dist


if __name__ == '__main__':
	# Setup environment
	barriers = [
		[[10, 15], [100, 10]],
		[[20, 30], [100, 40]]
	]
	
	# Setup droplet
	d = DropletAnimation(barriers)
	d.set_max_dist(100)
	num = d.find_shape(5000)
	#print 'Needed {} runs'.format(num)
	#assert d.get_area() == d.get_area_pure()

	# Plot the environment
	for p1, p2 in barriers:
		plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', linestyle='-', linewidth=2)

	# Plot the droplet
	x, y = d.get_max()
	scatter(x, y, color = 'red')

	x, y = d.get_shape()
	scatter(x, y, color = 'blue')

	axis('equal')
	show()