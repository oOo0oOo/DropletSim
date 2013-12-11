import math, random
import copy
import numpy as np
import pygame
from pygame.locals import QUIT, KEYDOWN, K_LEFT, K_RIGHT, K_UP, K_DOWN

pygame.init()
font = pygame.font.SysFont('helvetica', 25)

el_1 = 0.999
el_2 = 2-el_1

def intersect_lines(p1, p2, p3, p4):
	p43_x = (p3[0] - p4[0])
	p43_y = (p3[1] - p4[1])
	p21_x = (p1[0] - p2[0])
	p21_y = (p1[1] - p2[1])

	u = (p21_x * p43_y - p21_y * p43_x)
	
	if u > 0:
		a = (p3[0]*p4[1] - p3[1]*p4[0])
		b = (p1[0]*p2[1] - p1[1]*p2[0])

		x = (b * p43_x - p21_x * a) / u
		y = (b * p43_y - p21_y * a) / u

		# Check if on second line segment or outside
		# Not very versatile...
		# give it some elasticity
		if min(p3[0], p4[0]) * el_1 <= x <= max(p3[0], p4[0]) * el_2 and min(p3[1], p4[1]) * el_1 <= y <= max(p3[1], p4[1]) * el_2:
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

		self._center_point = np.array(start_point)
		self._barriers = np.array(copy.deepcopy(barriers))
		self._angle_list = np.array([i * self._angle for i in xrange(num_spikes)])

		self._outer_x = np.cos(self._angle_list) * self.max_len_spike
		self._outer_y = np.sin(self._angle_list) * self.max_len_spike
		self._outer = np.array(zip(self._outer_x, self._outer_y))

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

	def move_relax(self, t, ratio):
		cp = self.center_point
		s = self.stress_vector(ratio)
		self.move_center_point((s[0] - cp[0] + t[0], s[1] - cp[1] + t[1]))

	def relax(self, step_size = 0.5):
		c = self.center_point
		s = self.stress_vector(step_size)
		self.move_center_point((s[0] - c[0], s[1] - c[1]))

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


	def reset_max_dist_matrices(self):
		cp = np.array(self.center_point)

		# All the extreme points
		extreme_points = self._outer + cp

		for i, outer in enumerate(extreme_points):
			len_inter = self.max_len_spike

			for p1, p2 in self._barriers:
				# | x2-x1  x3-x1 |
				# | y2-y1  y3-y1 |
				matrix = np.vstack([p2-p1, outer-p1])
				orientation = np.linalg.det(matrix)

				if orientation < 0:
					first = cp
					second = outer
				else:
					first = outer
					second = cp

				da = second-first
				db = p2-p1
				dp = first-p1

				# Do dot products
				dap = np.empty_like(da)
				dap[0] = -da[1]
				dap[1] = da[0]
				denom = np.dot( dap, db)
				num = np.dot( dap, dp )

				if denom > 0 and num > 0:
					p_inter = (num / denom)*db + p1

					if min(p1[0], p2[0]) * el_1 <= p_inter[0] <= max(p1[0], p2[0]) * el_2 and min(p1[1], p2[1]) * el_1 <= p_inter[1] <= max(p1[1], p2[1]) * el_2:
						# calculate distance to intersection point
						vect = [p_inter[0] - cp[0], p_inter[1] - cp[1]]
						dist = math.sqrt((vect[0]**2) + (vect[1]**2))
						if dist < len_inter:
							len_inter = dist

			self._max_dist[i] = len_inter

	def reset_max_dist(self):
		cp = self.center_point

		for i in xrange(self.num_spikes):
			x = (self.max_len_spike * math.cos(i*self._angle)) + cp[0]
			y = (self.max_len_spike * math.sin(i*self._angle)) + cp[1]
			outer = (x, y)

			len_inter = self.max_len_spike
			inter = False

			for p1, p2 in self.barriers:
				#Check on which side of the line the point is
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

			self._max_dist[i] = len_inter

	def geometrical_center(self):
		x, y = self.get_shape()
		points = np.array([[x[i], y[i]] for i in xrange(self.num_spikes)])
		pos = np.sum(points, 0) * (1.0 / self.num_spikes) 
		return (pos[0], pos[1])

	def stress_vector(self, ratio = 1):
		centroid = self.geometrical_center()
		center = self.center_point
		vect = (centroid[0] - center[0], centroid[1] - center [1])
		x = float(center[0] + ratio * vect[0])
		y = float(center[1] + ratio * vect[1])
		return (x, y)

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

def intify(tup):
	return tuple([int(round(t)) for t in tup])

if __name__ == '__main__':
	# Setup environment

	rand = [random.randrange(0, 60) for i in range(25)]
		
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
	max_fps = 30

	d = DropletAnimation(barriers, start_point, num_spikes = num_spikes, max_dist = max_dist,area = area)
	d.move_center_point((0, 0))

	fpsClock = pygame.time.Clock()
	window = pygame.display.set_mode(size)
	pygame.display.set_caption('Droplet Simulation')

	white = pygame.Color(245, 245, 245)
	black = pygame.Color(10, 10, 10)
	blue = pygame.Color(10, 10, 245)
	red = pygame.Color(245, 10, 10)

	# set up the color scheme (3D Fade)
	radius = math.sqrt(area/math.pi)
	circumference = math.pi * 2 * radius
	relaxed = circumference / num_spikes
	rg = ColorGradient([0.5, relaxed * 2], (245, 50, 50), (50, 245, 50))

	# The animation loop
	while True:
		window.fill(white)

		# Draw all barriers
		for p1, p2 in barriers:
			pygame.draw.aaline(window, black, p1, p2, 1)

		# Draw center point, centroid & stress vector
		center = intify(d.center_point)
		pygame.draw.circle(window, blue, center, 3)
		#stress = intify(d.stress_vector())
		#pygame.draw.line(window, red, center, stress, 3)

		#Draw boundary
		x,y = d.get_shape()
		all_points = [[x[i], y[i]] for i in xrange(d.num_spikes)]
		for i, point in enumerate(all_points):
			next_point = all_points[(i + 1)%d.num_spikes]
			# get color according to scheme
			vect = [next_point[0] - point[0], next_point[1] - point[1]]
			dist = math.sqrt((vect[0]**2) + (vect[1]**2))
			c = intify(rg.get_color(dist))
			color =  pygame.Color(c[0], c[1], c[2])
			pygame.draw.line(window, color, point, next_point, 3)


		#Show statistics
		c = d.center_point
		s = d.stress_vector(1)
		vect = round(math.sqrt((s[0] - c[0]) ** 2 + (s[1] - c[1]) ** 2), 1)
		text = 'Speed: ' + str(int(fpsClock.get_fps())) + ' fps    Stress: ' + str(vect)
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
			d.move_relax(direction, 0.3)
		else:
			d.move_center_point(direction)


		fpsClock.tick(max_fps)