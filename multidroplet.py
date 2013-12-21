#!/usr/bin/env python

import math, random
import numpy as np
from string import Template

import pygame
from pygame.locals import QUIT, KEYDOWN, K_LEFT, K_RIGHT, K_UP, K_DOWN

# The PyCUDA stuff
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule


c_source = r'''

/* Some global variables */
__device__ float x_coords[$num_spikes], y_coords[$num_spikes];
__device__ float center_x[$num_droplets], center_y[$num_droplets];
__device__ int per_droplet, total_number, num_droplets;
__device__ float angle, threshold, max_dist;

__global__ void init_memory(int per, int total, float ang, float thresh, float max_d){
	per_droplet = per;
	total_number = total;
	angle = ang;
	threshold = thresh;
	max_dist = max_d;

	num_droplets = total/per;
}

__global__ void check_stop(int *stopped, float *spikes, float *cen_x, float *cen_y)
{	
	
	/* Calculate points */

	const int i = blockDim.x*blockIdx.x + threadIdx.x;
	const int num = (i - (i % per_droplet)) / per_droplet;

	const float c_x = cen_x[num];
	const float c_y = cen_y[num];
	center_x[num] = c_x;
	center_y[num] = c_y;

	float dist, v_x, v_y;

	const float x_i = spikes[i] * cos(angle * i) + c_x;
	const float y_i = spikes[i] * sin(angle * i) + c_y;

	x_coords[i] = x_i;
	y_coords[i] = y_i;

	__syncthreads();

	/*
		Check all droplets with a larger id (check each spike only once)
			if other center within square of length 2 * max_dist around own center
				find closest spike

		If closer than threshold: stop spike
	*/

	for(int m = num + 1; m < num_droplets; m++) {
		if (abs(center_x[m] - c_x) < max_dist && abs(center_y[m] - c_y) < max_dist){
			for(int n = m * per_droplet; n < (m + 1) * per_droplet; n++) {
				v_x = x_coords[n] - x_i;
				v_y = y_coords[n] - y_i;
				dist = sqrt(v_x * v_x + v_y * v_y);
				if (dist < threshold){
					stopped[i] = 1;
					stopped[n] = 1;
				};
			};
		};
		
	};
}
'''

# The Cython speedup
from c_code import intersect_lines as intersect
from c_code import distance, find_max_dist, get_areas

class DropletAnimation(object):
	def __init__(self, barriers, centers = [[100, 100], [300, 600], [700, 100]], num_spikes = 50, area = 7000, max_dist = 100):
		self.num_spikes = num_spikes
		self.area = area
		self.max_len_spike = max_dist

		self._angle = math.radians(360. / self.num_spikes)
		self._mult_term = np.array([math.sin(self._angle) * 0.5])
		self._angles = [i*self._angle for i in xrange(self.num_spikes)]
		self._np_angles = np.array(self._angles)
		
		self._c_barriers = np.array([[float(a), float(b), float(c), float(d)] for (a,b), (c, d) in barriers])

		# The multi approach
		self._centers = np.array(centers).astype(np.float32)
		self.num_droplets = self._centers.shape[0]
		self._num_droplets = np.int32(self.num_droplets)
		self._total_spikes = np.int32(num_spikes * self.num_droplets)
		self._spikes = np.zeros(self._total_spikes).astype(np.float32)
		self._max_dist = np.copy(self._spikes)

		self._stopped = np.zeros_like(self._spikes).astype(np.int32)

		self._collision_threshold = np.float32(3.0)
		self._max_dist_droplet = np.float32(150.0)
		self.per_droplet = num_spikes
		self._per_droplet = np.int32(num_spikes)

		self._np_angles_complete = np.hstack([self._np_angles for i in range(self._num_droplets)])
		self._centers_x = np.repeat([i[0] for i in centers], self.num_spikes)
		self._centers_y = np.repeat([i[1] for i in centers], self.num_spikes)

		self._points = np.empty((self._total_spikes, 2), dtype = np.float32)

		# Setup the indices
		self._indices = []
		for i in range(self._num_droplets):
			start = i * self.num_spikes
			stop = ((i + 1) * self.num_spikes)
			self._indices.append((start, stop))

		# Setup pycuda stuff
		self._spikes_grid = (self.num_droplets,1)
		self._spikes_block = (self.per_droplet,1,1)

		# A little bit of dynamics... (allocate right amount of memory)
		s = Template(c_source).substitute(num_spikes = self._total_spikes,
			num_droplets = self.num_droplets)

		self.mod = SourceModule(s)
		self._check_stop_raw = self.mod.get_function("check_stop")
		self._init_memory_raw = self.mod.get_function("init_memory")

		# Setup the constants
		self._init_memory_raw(self._per_droplet, self._total_spikes, np.float32(self._angle),
				self._collision_threshold, self._max_dist_droplet,
				grid = (1,1), block = (1,1,1)
				)

		#check_stop_types = [np.ndarray, np.ndarray, np.ndarray, np.int32, np.int32]
		#self._check_stop_raw.prepare(check_stop_types)


	def get_areas(self):
		return get_areas(self._spikes, self._mult_term, self._per_droplet)

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

	def get_points(self):
		#self._points = get_points(self._points, self._centers, self._spikes, self._per_droplet, self._angle)
		x = self._spikes * np.cos(self._np_angles_complete) + self._centers_x
		y = self._spikes * np.sin(self._np_angles_complete) + self._centers_y
		return x.astype(np.float32), y.astype(np.float32)

	def find_shapes(self):
		self._spikes[:] = 0.00001
		self._stopped[:] = 0

		areas = self.get_areas()
		num = 0
		n = self._total_spikes

		min_val = 0.008*n

		# Setup center points for pycuda
		c_x = drv.In(self._centers[:, 0].astype(np.float32))
		c_y = drv.In(self._centers[:, 1].astype(np.float32))

		while n > min_val and num < 150:
			n = self.move_up_spikes(areas)
			areas = self.get_areas()

			# check for collision with other droplet (pycuda)
			self._check_stop_raw(
				drv.InOut(self._stopped), drv.In(self._spikes.astype(np.float32)), 
				c_x, c_y, grid = self._spikes_grid, block = self._spikes_block,
				)

			num += 1

	def reset_max_dist(self):
		'''
			Changed to use cython optimized code.
		'''
		self._max_dist[:] = 0.0 
		for num, (start, stop) in enumerate(self._indices):
			c = self._centers[num]
			self._max_dist[start:stop] = find_max_dist(self._c_barriers, c[0], c[1], self.max_len_spike, self._np_angles)

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
	num_spikes = 100, max_dist = 200, max_fps = 500, num_frames = -1, capture_folder = ''):

	if capture_folder != '':
		import os
		if not os.path.exists(capture_folder):
			os.makedirs(capture_folder)
	
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
	frame = 0
	while num_frames == -1 or frame < num_frames:
		# Update agents
		d.move_relax(direction, relax)

		window.fill(white)

		# Draw all barriers
		for p1, p2 in barriers:
			pygame.draw.aaline(window, black, p1, p2, 1)

		# Draw center points, centroid & stress vector
		for x, y in d._centers:
			pygame.draw.circle(window, blue, (int(x), int(y)), 3)

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

		frame += 1

		if capture_folder != '':
			pygame.image.save(window, '{}/{}.jpg'.format(capture_folder, frame))

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
	direction = (2, 0)
	relax = 1.5
	area = 7000.
	num_spikes = 50
	max_dist = 200
	max_fps = 500
	num_droplets = 50
	n = -1

	centers = [[-random.randrange(0, 20*num_droplets), random.randrange(80, size[1]-80)] for i in range(num_droplets)]

	start_simulation(size, barriers, centers, area, direction = direction, num_frames = n)