# Import multidroplet simulation
import sys, random

sys.path.append("..")
from multidroplet import start_simulation

size = (800, 800)

barriers = [
	# The outer boundaries
	[[20, 20], [20, size[1]-20]],
	[[20, size[1]-20], [size[0]-20, size[1]-20]],
	[[size[0]-20, size[1]-20], [size[0]-20, 0]],
	[[size[0]-20, 20], [20, 20]]
]

#start_point = (size[0]/8, 200)
direction = (0, 0)
relax = 3.0
area = 8200.
num_spikes = 80
max_dist = 500
max_fps = 500
num_droplets = 64

centers = [[random.randrange(size[0] * 0.1, size[0] * 0.9), random.randrange(size[1] * 0.1, size[1] * 0.9)] for i in range(num_droplets)]

start_simulation(size, barriers, centers, area, direction, 
	relax, num_spikes, max_dist, max_fps)