# Import multidroplet simulation
import sys, random

sys.path.append("..")
from multidroplet import start_simulation

size = (1400, 980)

mid = (size[0]/2, size[1]/2)
barriers = [
	# The outer boundaries
	[[20, 20], [20, size[1]-20]],
	[[20, size[1]-20], [size[0]-20, size[1]-20]],
	[[size[0]-20, size[1]-20], [size[0]-20, 0]],
	#[[size[0]-20, 20], [20, 20]]

]


def christmas_tree():
	s_width = 200
	s_height = 150
	b = []

	# Stem
	b.append([[mid[0] - s_width/2, 800], [mid[0] - s_width/2, 800 - s_height]])
	b.append([[mid[0] + s_width/2, 800], [mid[0] + s_width/2, 800 - s_height]])

	# Spikes
	for i in range(3):
		n = 350 - 75 * i
		#b.append([[mid[0] - s_width/2, 800 - s_height - i*200], [mid[0] - n, 800 - s_height - i*200]])
		b.append([[mid[0] - n, 800 - s_height - i*200], [mid[0] - n + 200, 800 - s_height - (i+1)*200]])

		#b.append([[mid[0] + s_width/2, 800 - s_height - i*200], [mid[0] + n, 800 - s_height - i*200]])
		b.append([[mid[0] + n, 800 - s_height - i*200], [mid[0] + n - 200, 800 - s_height - (i+1)*200]])
	
	return b

barriers = barriers + christmas_tree()


#start_point = (size[0]/8, 200)
direction = (0, 2)
relax = 3.0
area = 6500.
num_spikes = 80
max_dist = size[0]/2
max_fps = 500
num_droplets = 300
num_frames = 2000
folder = 'christmas'
#folder = ''

centers = [[random.randrange(100*size[0] * 0.1, 100*size[0] * 0.9)/100, -1.5 * random.randrange(100*size[1] * 0.1, 100*size[1] * 0.9)/100] for i in range(num_droplets)]

start_simulation(size, barriers, centers, area, direction, 
	relax, num_spikes, max_dist, max_fps, num_frames, folder)