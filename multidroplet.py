#!/usr/bin/env python

import math, random
import numpy as np

import pygame
from pygame.locals import QUIT, KEYDOWN, K_LEFT, K_RIGHT, K_UP, K_DOWN

# The PyCUDA stuff
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule


c_source = r'''
__global__ void check_stop(int *stopped, float *x_coords, float *y_coords, int per_droplet, int total_number)
{
	const int i = blockDim.x*blockIdx.x + threadIdx.x;
	const float threshold = 3.5;

	/* Loop through all running points */
	if (stopped[i] == 0){

		/* Find start_ignore and end_ignore  (own droplets spikes) */
		const int num = (i - (i % per_droplet)) / per_droplet;
		const int start_ignore = num * per_droplet;
		const int end_ignore = (num + 1) * per_droplet;

		float distance, min_dist, v_x, v_y, x_i, y_i;
		x_i = x_coords[i];
		y_i = y_coords[i];
		min_dist = 1000000.0;

		for(int n = 0; n < start_ignore; n++) {
			v_x = x_coords[n] - x_i;
			v_y = y_coords[n] - y_i;

			distance = sqrt(v_x * v_x + v_y * v_y);
			if (distance < min_dist){
				min_dist = distance;
			};
		
		};

		for(int n = end_ignore; n < total_number; n++) {
			v_x = x_coords[n] - x_i;
			v_y = y_coords[n] - y_i;
			distance = sqrt(v_x * v_x + v_y * v_y);
			if (distance < min_dist){
				min_dist = distance;
			};

		};

		if (min_dist < threshold) {
			stopped[i] = 1;
		}
		else {
			stopped[i] = 0;
		}

	}
}
'''

# The Cython speedup
from c_code import intersect_lines as intersect
from c_code import distance, find_max_dist, downsample, find_shape_multiple

class DropletAnimation(object):
	def __init__(self, barriers, centers = [[100, 100], [300, 600], [700, 100]], num_spikes = 50, area = 7000, max_dist = 100):
		self.barriers = barriers
		self.num_spikes = num_spikes
		self.area = area
		self.max_len_spike = max_dist

		self._angle = math.radians(360. / self.num_spikes)
		self._mult_term = np.array([math.sin(self._angle) * 0.5])
		self._angles = [i*self._angle for i in xrange(self.num_spikes)]
		self._np_angles = np.array(self._angles).astype(np.float32)
		
		self._c_barriers = [[float(a), float(b), float(c), float(d)] for (a,b), (c, d) in barriers]

		# The multi approach
		self._barriers = np.array(self.barriers).astype(np.float32)
		self._centers = np.array(centers).astype(np.float32)
		self.num_droplets = self._centers.shape[0]
		self._num_droplets = np.int32(self.num_droplets)
		self._total_spikes = np.int32(num_spikes * self.num_droplets)
		self._spikes = np.zeros(self._total_spikes).astype(np.float32)
		self._max_dist = np.copy(self._spikes)

		self._stopped = np.zeros_like(self._spikes).astype(np.int32)

		self._collision_threshold = np.float32(3.0)
		self.per_droplet = num_spikes
		self._per_droplet = np.int32(num_spikes)

		self._np_angles_complete = np.hstack([self._np_angles for i in range(self._num_droplets)])
		self._centers_x = np.repeat([i[0] for i in centers], self.num_spikes)
		self._centers_y = np.repeat([i[1] for i in centers], self.num_spikes)

		self.next_relax = np.array([[0.0, 0.0] for i in range(self._num_droplets)])

		# Setup the indices
		self._indices = []
		for i in range(self._num_droplets):
			start = i * self.num_spikes
			stop = ((i + 1) * self.num_spikes)
			self._indices.append((start, stop))

		# Setup pycuda stuff
		self.mod = SourceModule(c_source)
		self._check_stop_raw = self.mod.get_function("check_stop")

	def get_areas(self):
		areas = []
		for start, stop in self._indices:
			#extra = (stop+1)%self._total_spikes
			areas.append(np.sum(self._spikes[start:stop - 1] * self._spikes[start+1:stop] * self._mult_term))
		return areas

	def move_up_spikes(self, areas):
		cond1 = self._spikes < self._max_dist
		cond2 = self._stopped == 0
		cond = np.logical_and(cond1, cond2)

		steps = np.array([(2.5 * (self.area - a) / self.area) if a <= self.area else 0.0 for a in areas])
		steps[np.where(np.logical_and(steps>0, steps<0.5))] = 0.5

		steps = np.repeat(steps, self.num_spikes)
		self._spikes[cond] += steps[cond]

		return np.sum(steps[cond])

	def get_shapes(self):
		shapes = []
		for num, (start, stop) in enumerate(self._indices):
			x = self._spikes[start:stop] * np.cos(self._np_angles) + self._centers[num][0]
			y = self._spikes[start:stop] * np.sin(self._np_angles) + self._centers[num][1]
			shapes.append((x, y))

		return shapes

	def get_max(self):
		shapes = []
		for num, (start, stop) in enumerate(self._indices):
			x = self._max_dist[start:stop] * np.cos(self._np_angles) + self._centers[num][0]
			y = self._max_dist[start:stop] * np.sin(self._np_angles) + self._centers[num][1]
			shapes.append((x, y))
		return shapes

	def get_points(self):
		x = self._spikes * np.cos(self._np_angles_complete) + self._centers_x
		y = self._spikes * np.sin(self._np_angles_complete) + self._centers_y
		return x.astype(np.float32), y.astype(np.float32)

	def find_shapes(self):
		self._spikes[:] = 0.00001
		self._stopped[:] = 0

		areas = self.get_areas()
		num = 0
		n = self._total_spikes

		min_val = 0.01*n

		while n > min_val and num < 150:
			n = self.move_up_spikes(areas)
			areas = self.get_areas()

			# check for collision with other droplet
			x, y = self.get_points()
			self._check_stop_raw(drv.InOut(self._stopped), drv.In(x), drv.In(y), 
				self._per_droplet, self._total_spikes, 
				grid=(self.per_droplet,1), block=(self.num_droplets,1,1))

			num += 1

	def reset_max_dist(self):
		'''
			Changed to use cython optimized code.
		'''
		self._max_dist[:] = 0.0 
		for num, (start, stop) in enumerate(self._indices):
			c = self._centers[num]
			self._max_dist[start:stop] = find_max_dist(self._c_barriers, c[0], c[1], self.max_len_spike, self._angles)

	def geometrical_centers(self):
		c = []
		fact = (1.0 / self.num_spikes)
		shapes = self.get_shapes()
		for x, y in shapes:
			x_coord = np.sum(x) * fact
			y_coord = np.sum(y) * fact
			c.append( (x_coord, y_coord) )
		return c

	def stress_vectors(self, ratio = 1):
		vectors = []
		centroids = self.geometrical_centers()
		for i, c in enumerate(centroids):
			cent = self._centers[i]
			vect = (c[0] - cent[0], c[1] - cent[1])
			x = float(ratio * vect[0])
			y = float(ratio * vect[1])
			vectors.append( [x, y] )
		stress = np.array(vectors).astype(np.float32)
		return stress # + np.random.uniform(-1, 1, stress.shape) * 0.1


	def move_relax(self, t, ratio):
		self._centers += t
		self._centers += self.stress_vectors(ratio)

		self._centers_x = np.repeat(self._centers[:, 0], self.num_spikes)
		self._centers_y = np.repeat(self._centers[:, 1], self.num_spikes)

		self.reset_max_dist()
		self.find_shapes()


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

def start_simulation(size, barriers, centers, area, direction = (3, 0), relax = 1.0,
	num_spikes = 100, max_dist = 200, max_fps = 500):
	
	pygame.init()
	font = pygame.font.SysFont('helvetica', 25)

	fpsClock = pygame.time.Clock()
	window = pygame.display.set_mode(size)
	pygame.display.set_caption('Droplet Simulation')

	white = pygame.Color(245, 245, 245)
	black = pygame.Color(10, 10, 10)
	blue = pygame.Color(10, 10, 245)
	red = pygame.Color(245, 10, 10)

	d = DropletAnimation(barriers = barriers, centers = centers, num_spikes = num_spikes, 
		max_dist = max_dist, area = area)

	# The animation loop
	while True:
		# Update agents
		d.move_relax(direction, relax)

		window.fill(white)

		# Draw all barriers
		for p1, p2 in barriers:
			pygame.draw.aaline(window, black, p1, p2, 1)

		# Draw center points, centroid & stress vector
		#for x, y in d._centers:
			#pygame.draw.circle(window, blue, (int(x), int(y)), 3)

		#Draw boundary
		shapes = d.get_shapes()
		areas = d.get_areas()

		num = len(shapes[0][0])
		for j, (x, y) in enumerate(shapes):
			if areas[j] * 1.1 >= area:
				color = blue
			else:
				color = red

			for i in xrange(num):
				point = (x[i], y[i])
				next_i = (i + 1)%num
				next_point = (x[next_i], y[next_i])
				pygame.draw.aaline(window, color, point, next_point, 2)

		#Show statistics
		#s = d.stress_vector(1)
		#vect = round(math.sqrt((s[0] - c[0]) ** 2 + (s[1] - c[1]) ** 2), 1)
		text = str(int(fpsClock.get_fps())) + ' fps'
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

		fpsClock.tick(max_fps)


if __name__ == '__main__':
	size = (1200, 400)

	barriers = [
		# Some channel walls
		[[50, size[1] - 50], [200, 280]],
		[[50, 50], [200, 120]],

		# Two layer
		[[200, 280], [450, 280]],
		[[200, 120], [450, 120]],

		# Constriction
		[[450, 280], [600, 240]],
		[[450, 120], [600, 160]],

		# Alternation
		[[600, 240], [1000, 250]],
		[[600, 160], [1000, 150]],
	]

	#start_point = (size[0]/8, 200)
	direction = (3, 0)
	relax = 1.5
	area = 7000.
	num_spikes = 80
	max_dist = 200
	max_fps = 500
	num_droplets = 15

	centers = [[-random.randrange(0, 20*num_droplets), random.randrange(80, size[1]-80)] for i in range(num_droplets)]

	start_simulation(size, barriers, centers, area)
	
