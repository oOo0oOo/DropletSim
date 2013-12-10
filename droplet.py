import numpy as np
import math

class Droplet(object):
	def __init__(self):
		self.center_point = (100, 100)
		self.num_spikes = 3

		self._spikes = np.array([0.1 for i in range(self.num_spikes)])
		self._angle = 360. / self.num_spikes
		self._spike_indices = range(len(self._spikes))
		self.mult_term = math.sin(self._angle) * 0.5 

	def get_diameter(self):
		return 2 * sum(self._spikes)/self.num_spikes

	def get_area(self):
		mult = np.copy(self._spikes)
		mult[[0, -1]] = mult[[-1, 0]]
		area = np.sum(self._spikes * mult * self.mult_term)
		return area

	def get_area_old(self):
		area = 0
		for i, p in enumerate(self._spikes):
			area += p * self._spikes[(i+1)%self.num_spikes] * self.mult_term
		return area

	def move_up_spikes(self, distance):
		self._spikes += distance

	def get_points(self):
		points = []
		for i, p in enumerate(self._spikes):
			x = (p * math.cos(i*self._angle)) + self.center_point[0]
			y = (p * math.sin(i*self._angle)) + self.center_point[1]
			points.append((x, y))
		return points

	def find_shape(self, area):
		v = d.get_area()
		while v < area:
			d.move_up_spikes(0.1)
			v = d.get_area_old()


d = Droplet()
d.find_shape(10000000)
#print d.get_area()
#print d.get_area_old()
# print d.get_points()