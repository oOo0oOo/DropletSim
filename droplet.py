import numpy as np
import math
from pylab import scatter, axis, show, plot

class Droplet(object):
	def __init__(self):
		self.center_point = (50, 25)
		self.num_spikes = 50

		self._spikes = np.array([0.1 for i in xrange(self.num_spikes)])
		self._angle = 360. / self.num_spikes
		self._spike_indices = xrange(self.num_spikes)
		self._mult_term = math.sin(self._angle) * 0.5 

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
		self._spikes += distance

	def get_points(self):
		x, y = [], []
		for i, p in enumerate(self._spikes):
			x.append((p * math.cos(i*self._angle)) + self.center_point[0])
			y.append((p * math.sin(i*self._angle)) + self.center_point[1])
		return x, y

	def find_shape(self, area):
		v = d.get_area()
		n  = 0
		while v < area:
			n += 1
			d.move_up_spikes(0.05)
			v = d.get_area()
		return n

class Environment(object):
	def __init__(self, lines):
		self._lines = np.array(lines)

	def closest_intersect(self):
		for p1, p2 in self._lines:
			x = (p1[0]*p2[1] - p1[1]*p2[0])*(p3[0] - p4[0]) - (p1[0] - p2[0])*(p3[0]*p4[1] - p3[1]*p4[0])
			x /= ((p1[0] - p2[0])*(p3[1] - p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0]))

			y = ((p1[0]*p2[1] - p1[1]*p2[0])*(p3[1] - p4[1])) - ((p1[1] - p2[1])*(p3[0]*p4[1] - p3[1]*p4[0]))
			y /= ((p1[0] - p2[0])*(p3[1] - p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0]))

			print x, y



if __name__ == '__main__':
	# Setup environment
	lines = [
		[[10, 10], [100, 10]],
		[[10, 30], [100, 40]],

	]
	
	e = Environment(lines)

	# Setup droplet
	d = Droplet()
	num = d.find_shape(5000)
	#print 'Needed {} runs'.format(num)
	#assert d.get_area() == d.get_area_pure()

	# Plot the environment
	for p1, p2 in e._lines:
		plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', linestyle='-', linewidth=2)

	# Plot the droplet
	x, y = d.get_points()
	scatter(x, y)

	axis('equal')
	show()