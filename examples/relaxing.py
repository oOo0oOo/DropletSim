# Import multidroplet simulation
import sys, random

sys.path.append("..")
from multidroplet import start_simulation

size = (1800, 980)

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
max_dist = size[0]/2
max_fps = 500
num_droplets = 375
num_frames = 200
#folder = 'relaxing'
folder = ''

centers = [[random.randrange(100*size[0] * 0.1, 100*size[0] * 0.9)/100, random.randrange(100*size[1] * 0.1, 100*size[1] * 0.9)/100] for i in range(num_droplets)]

start_simulation(size, barriers, centers, area, direction, 
	relax, num_spikes, max_dist, max_fps, num_frames, folder)