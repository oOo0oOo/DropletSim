import math, random
import numpy as np
import pygame
from pygame.locals import QUIT, KEYDOWN, K_LEFT, K_RIGHT, K_UP, K_DOWN


# The C speedup
from c_code import intersect_lines as intersect
from c_code import distance, find_max_dist, downsample

#from shapely.geometry import LineString, Point, MultiLineString

pygame.init()
font = pygame.font.SysFont('helvetica', 25)

class DropletAnimation(object):
	def __init__(self, barriers, start_point = (50, 30), num_spikes = 50, area = 7000, max_dist = 100):
		self.center_point = start_point
		self.barriers = barriers
		self.num_spikes = num_spikes
		self.area = area
		self.max_len_spike = max_dist

		self._spikes = np.array([0.1 for i in xrange(self.num_spikes)])
		self._angle = math.radians(360. / self.num_spikes)
		self._mult_term = np.array([math.sin(self._angle) * 0.5])
		self._max_dist = np.copy(self._spikes)
		self._downsampled = np.array(range(self.num_spikes))
		self._angles = [i*self._angle for i in xrange(self.num_spikes)]
		self._np_angles = np.array(self._angles)
		self._c_barriers = [[float(a), float(b), float(c), float(d)] for (a,b), (c, d) in barriers]

	def get_average_radius(self):
		return np.average(self._spikes)

	def get_area(self):
		return np.sum(self._spikes * np.roll(self._spikes, 1) * self._mult_term)

	def get_max_area(self):
		return np.sum(self._max_dist * np.roll(self._max_dist, 1) * self._mult_term)

	def move_up_spikes(self, distance):
		self._spikes[self._spikes < self._max_dist] += distance

	def move_center_point(self, t, sample_rate = 0.5):
		cp = self.center_point
		self.center_point = (cp[0] + t[0], cp[1] + t[1])
		self.reset_max_dist()
		self.find_shape()
		self.downsample(sample_rate)

	def move_relax(self, t, ratio, sample_rate):
		cp = self.center_point
		s = self.stress_vector(ratio)
		self.move_center_point((s[0] - cp[0] + t[0], s[1] - cp[1] + t[1]), sample_rate)

	def relax(self, step_size = 0.5):
		c = self.center_point
		s = self.stress_vector(step_size)
		self.move_center_point((s[0] - c[0], s[1] - c[1]))

	def get_shape(self):
		x = self._spikes * np.cos(self._np_angles) + self.center_point[0]
		y = self._spikes * np.sin(self._np_angles) + self.center_point[1]
		return x, y

	def get_max(self):
		x = self._max_dist * np.cos(self._np_angles) + self.center_point[0]
		y = self._max_dist * np.sin(self._np_angles) + self.center_point[1]
		return x, y

	def get_downsampled(self):
		ds = self._downsampled
		x = self._spikes[ds] * np.cos(self._np_angles[ds]) + self.center_point[0]
		y = self._spikes[ds] * np.sin(self._np_angles[ds]) + self.center_point[1]
		return x, y

	def find_shape(self, point = False):
		self._spikes[:] = 0.1

		if point and type(point) == tuple and len(point) == 2:
			self.center_point = point

		if self.area < self.get_max_area():
			v = self.get_area()
			while v < self.area:
				step = (self.area - v) / self.area
				if step < 0.3: step = 0.3
				self.move_up_spikes(step)
				v = self.get_area()
		else:
			print 'Area too large!!'
			self._spikes = np.copy(self._max_dist)

	def downsample(self, sample_rate):
		points = np.dstack(self.get_shape())[0]
		self._downsampled = downsample(points, sample_rate)

	def reset_max_dist(self):
		'''
			Changed to use cython optimized code.
		'''
		cp_x, cp_y = self.center_point
		self._max_dist[:] = find_max_dist(self._c_barriers, cp_x, cp_y, self.max_len_spike, self._angles)

	def geometrical_center(self):
		x, y = self.get_shape()
		fact = (1.0 / self.num_spikes)
		x = np.sum(x) * fact
		y = np.sum(y) * fact
		return (x, y)

	def stress_vector(self, ratio = 1):
		centroid = self.geometrical_center()
		center = self.center_point
		vect = (centroid[0] - center[0], centroid[1] - center [1])
		x = float(center[0] + ratio * vect[0])
		y = float(center[1] + ratio * vect[1])
		return (x, y)

class ColorGradient(object):
	def __init__(self, extremes = [5, 100], low_color = [245, 10, 10], high_color = [10, 245, 10], shades = 100):
		self.low = low_color
		self.high = high_color
		self.extremes = extremes
		self.shades = shades

		# The increment
		absolut = [self.high[0] - self.low[0], self.high[1] - self.low[1], self.high[2] -self.low[2]]
		self._per_unit = [float(a) / self.shades for a in absolut]
		self.sum_extremes = sum(extremes)

		# The complete gradient (pre calculated)
		self.color = {}
		for j in range(shades):
			self.color[j] = [int(self.low[i] + self._per_unit[i] * j) for i in range(3)]

	
	def get_color(self, value):
		if value <= self.extremes[0]:
			return self.low
		elif value >= self.extremes[1]:
			return self.high
		else:
			fact = int(self.shades * (value - self.extremes[0]) / self.sum_extremes)
			return self.color[fact]

def intify(tup):
	return [int(tup[0]), int(tup[1])]

if __name__ == '__main__':
	# Setup environment

	rand = [random.randrange(10, 60) for i in range(25)]
		
	barriers = [
		# Some channel walls
		[[100, 200 + rand[1]], [200, 200 + rand[2]]],
		[[100, 200 - rand[11]], [200, 200 - rand[12]]],

		[[200, 200 + rand[2]], [300, 200 + rand[3]]],
		[[200, 200 - rand[12]], [300, 200 - rand[13]]],

		[[300, 200 + rand[3]], [400, 200 + rand[4]]],
		[[300, 200 - rand[13]], [400, 200 - rand[14]]],

		[[400, 200 + rand[4]], [500, 200 + rand[5]]],
		[[400, 200 - rand[14]], [500, 200 - rand[15]]],

		[[500, 200 + rand[5]], [600, 200 + rand[6]]],
		[[500, 200 - rand[15]], [600, 200 - rand[16]]],

		[[600, 200 + rand[6]], [700, 200 + rand[7]]],
		[[600, 200 - rand[16]], [700, 200 - rand[17]]],

		[[700, 200 + rand[7]], [800, 200 + rand[8]]],
		[[700, 200 - rand[17]], [800, 200 - rand[18]]],

		[[800, 200 + rand[8]], [900, 200 + rand[9]]],
		[[800, 200 - rand[18]], [900, 200 - rand[19]]],

		[[900, 200 + rand[9]], [1000, 200 + rand[10]]],
		[[900, 200 - rand[19]], [1000, 200 - rand[20]]],
	]

	size = (1200, 400)
	start_point = (size[0]/8, 200)
	direction = (5, 0)
	area = 8000.
	num_spikes = 250
	max_dist = 200
	max_fps = 500
	downsample_rate = 0.25 # Keep this percentage of edge points

	fpsClock = pygame.time.Clock()
	window = pygame.display.set_mode(size)
	pygame.display.set_caption('Droplet Simulation')

	white = pygame.Color(245, 245, 245)
	black = pygame.Color(10, 10, 10)
	blue = pygame.Color(10, 10, 245)
	red = pygame.Color(245, 10, 10)

	# set up the color scheme (3D Fade) & downsampling
	radius = math.sqrt(area/math.pi)
	circumference = math.pi * 2 * radius
	relaxed = circumference / num_spikes
	# rg = ColorGradient([0.5, relaxed * 2], (245, 50, 50), (50, 245, 50))

	d = DropletAnimation(barriers, start_point, num_spikes = num_spikes, max_dist = max_dist,area = area)
	d.move_center_point((0, 0), downsample_rate)

	# The animation loop
	while True:
		window.fill(white)

		# Draw all barriers
		for p1, p2 in barriers:
			pygame.draw.aaline(window, black, p1, p2, 1)

		# Draw center point, centroid & stress vector
		center = intify(d.center_point)
		pygame.draw.circle(window, blue, center, 3)

		#Draw boundary
		x,y = d.get_downsampled()
		num = len(x)
		for i in xrange(num):
			point = (x[i], y[i])
			next_i = (i + 1)%num
			next_point = (x[next_i], y[next_i])
			# get color according to scheme
			# dist = distance(point[0], point[1], next_point[0], next_point[1])
			#c = rg.get_color(dist)
			# color = pygame.Color(c[0], c[1], c[2])
			pygame.draw.aaline(window, blue, point, next_point, 1)


		#Show statistics
		c = d.center_point
		#s = d.stress_vector(1)
		#vect = round(math.sqrt((s[0] - c[0]) ** 2 + (s[1] - c[1]) ** 2), 1)
		text = 'Speed: ' + str(int(fpsClock.get_fps())) + ' fps    Stress: '# + str(vect)
		label = font.render(text, 1, black)
		rect = label.get_rect()
		rect.center = (size[0]/2, 20)
		window.blit(label, rect)

		#Handle events (single press, not hold)
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				break

		pygame.display.update()

		#Move it
		# Check for pressed leaning keys
		keys = pygame.key.get_pressed()
		not_here = False
		if keys[K_LEFT]:
			direction = (-5, 0)
		if keys[K_RIGHT]:
			direction = (5, 0)
		if keys[K_UP]:
			direction = (0, -5)
		if keys[K_DOWN]:
			direction = (0, 5)
		else:
			not_here = True

		if not_here:
			d.move_relax(direction, 0.3, downsample_rate)
		else:
			d.move_center_point(direction, downsample_rate)

		fpsClock.tick(max_fps)