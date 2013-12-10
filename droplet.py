import math, time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import pygame
from pygame.locals import QUIT

pygame.init()
font = pygame.font.SysFont('helvetica', 25)

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
				d.move_up_spikes(0.5)
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

class ColorGradient(object):
	def __init__(self, extremes = [5, 100],
		low_color = [245, 10, 10], high_color = [10, 245, 10]):

		self.extremes = extremes
		self.low = np.array(low_color)
		self.high = np.array(high_color)

		# The increment
		absolut = self.high - self.low
		diff = self.extremes[1] - self.extremes[0]
		self._per_unit = absolut / diff
	
	def get_color(self, value):
		if value <= self.extremes[0]:
			return self.low
		elif value >= self.extremes[1]:
			return self.high
		else:
			corr = value - self.extremes[0]
			return np.round(self.low + (self._per_unit * corr))



if __name__ == '__main__':
	# Setup environment
	def rand(val):
		return val + random.randrange(10, 70)
		
	barriers = [
		# The entry channel
		[[-100, 200], [100, 200]],
		[[-100, 150], [100, 150]],

		# The expansion
		[[100, 200], [200, 250]],
		[[100, 150], [200, 100]],

		# The constriction
		[[200, 250], [300, 200]],
		[[200, 100], [300, 150]],
		

		[[300, 200], [400, 220]],
		[[300, 150], [400, 130]],

		[[400, 220], [500, 190]],
		[[400, 130], [500, 160]],

		[[500, 190], [600, 210]],
		[[500, 160], [600, 140]],

		[[600, 210], [750, 230]],
		[[600, 140], [750, 120]],

		[[750, 230], [900, 200]],
		[[750, 120], [900, 150]],

		[[900, 200], [1050, 300]],
		[[900, 150], [1050, 50]],

		[[1050, 300], [1250, 200]],
		[[1050, 50], [1250, 150]]

	]

	start_point = (-50, 175)
	direction = (4,0)
	size = (1200, 400)
	area = 8000.
	num_spikes = 100
	max_dist = 200


	d = DropletAnimation(barriers, start_point, num_spikes = num_spikes, max_dist = max_dist,
			area = area)

	max_fps = 40
	num_frames = 200
	fpsClock = pygame.time.Clock()
	window = pygame.display.set_mode(size)
	pygame.display.set_caption('Droplet Simulation')

	white = pygame.Color(245, 245, 245)
	black = pygame.Color(10, 10, 10)
	blue = pygame.Color(10, 10, 245)

	# set up the color scheme (3D Fade)
	radius = math.sqrt(area/math.pi)
	circumference = math.pi * 2 * radius
	relaxed = circumference / num_spikes
	rg = ColorGradient([0, relaxed * 2])

	# The animation loop
	i=0
	while i < num_frames:
		i += 1
		window.fill(white)

		# Draw all barriers
		for p1, p2 in barriers:
			pygame.draw.aaline(window, black, p1, p2, 1)


		x,y = d.get_shape()
		all_points = [[x[i], y[i]] for i in xrange(d.num_spikes)]
		for i, point in enumerate(all_points):
			next_point = all_points[(i + 1)%d.num_spikes]
			# get color according to scheme
			vect = [next_point[0] - point[0], next_point[1] - point[1]]
			dist = math.sqrt((vect[0]**2) + (vect[1]**2))
			c = rg.get_color(dist)
			color =  pygame.Color(int(c[0]), int(c[1]), int(c[2]))
			pygame.draw.line(window, color, point, next_point, 3)


		#Show fps
		label = font.render(str(int(fpsClock.get_fps())), 1, black)
		rect = label.get_rect()
		rect.center = (20, 20)
		window.blit(label, rect)

		#Handle events (single press, not hold)
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				break

		pygame.display.update()

		# Update state of all objects

		if d.center_point[0] > size[0] + 100:
			direction = (-4,0)
		elif d.center_point[0] < -100:
			direction = (4,0)

		d.move_center_point(direction)

		fpsClock.tick(max_fps)