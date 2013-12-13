import random
from droplet import *
import time

rand = [48, 10, 34, 17, 42, 32, 20, 10, 28, 49, 41, 31, 27, 48, 42, 48, 47, 37, 13, 31, 51, 22, 45, 36, 17, 59, 45, 54, 42, 34]

		
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
start_point = (0, 200)

area = 8000.
num_spikes = 250
max_dist = 200

d = DropletAnimation(barriers, start_point, num_spikes = num_spikes, max_dist = max_dist, area = area)

def run_across_screen():
	'''
		Returns framerate and total number of frames
	'''
	before = time.time()
	n = 0
	while d.center_point[0] < size[0]:
		d.move_relax((1,0), 0.1, 0.5)
		n += 1

	return n / (time.time() - before), n
d.move_relax((1,0), 0.1, 0.5)
for i in xrange(0):
	fps, num = run_across_screen()
	print 'Calculated {} frames @ {} fps.'.format(num, fps)
	d.center_point = (0, 200)