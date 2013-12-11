import random
from droplet import *

rand = [random.randrange(12, 60) for i in range(15)]
		
barriers = [
	# Some channel walls
	[[-100, 200 + rand[0]], [100, 200 + rand[1]]],
	[[-100, 200 - rand[0]], [100, 200 - rand[1]]],

	[[100, 200 + rand[1]], [200, 200 + rand[2]]],
	[[100, 200 - rand[1]], [200, 200 - rand[2]]],

	[[200, 200 + rand[2]], [300, 200 + rand[3]]],
	[[200, 200 - rand[2]], [300, 200 - rand[3]]],

	[[300, 200 + rand[3]], [400, 200 + rand[4]]],
	[[300, 200 - rand[3]], [400, 200 - rand[4]]],

	[[400, 200 + rand[4]], [500, 200 + rand[5]]],
	[[400, 200 - rand[4]], [500, 200 - rand[5]]],

	[[500, 200 + rand[5]], [600, 200 + rand[6]]],
	[[500, 200 - rand[5]], [600, 200 - rand[6]]],

	[[600, 200 + rand[6]], [700, 200 + rand[7]]],
	[[600, 200 - rand[6]], [700, 200 - rand[7]]],

	[[700, 200 + rand[7]], [800, 200 + rand[8]]],
	[[700, 200 - rand[7]], [800, 200 - rand[8]]],

	[[800, 200 + rand[8]], [900, 200 + rand[9]]],
	[[800, 200 - rand[8]], [900, 200 - rand[9]]],

	[[900, 200 + rand[9]], [1000, 200 + rand[10]]],
	[[900, 200 - rand[9]], [1000, 200 - rand[10]]],

	[[1000, 200 + rand[10]], [1200, 200 + rand[11]]],
	[[1000, 200 - rand[10]], [1200, 200 - rand[11]]]

]

size = (1200, 400)
start_point = (size[0]/2, 200)
area = 8000.
num_spikes = 250
max_dist = 200
max_fps = 150

d = DropletAnimation(barriers, start_point, num_spikes = num_spikes, max_dist = max_dist, area = area)

for i in xrange(500):
	d.move_center_point((1, 0))
	d.relax()