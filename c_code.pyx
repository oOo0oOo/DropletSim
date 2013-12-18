cimport numpy as np
import numpy

# Inline min and max functions
cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b

# sqrt, cos and sin from stdlib
cdef extern from "stdlib.h":
	double c_libc_sqrt "sqrt"(double x)

cdef extern from "stdlib.h":
	double c_libc_cos "cos"(double x)

cdef extern from "stdlib.h":
	double c_libc_sin "sin"(double x)


#Elasticity parameters
cdef double el_1, el_2
el_1 = 0.999
el_2 = 2-el_1

cpdef tuple intersect_lines(double p1_x, double p1_y, double p2_x, double p2_y, double p3_x, double p3_y, double p4_x, double p4_y):
	# Declare all the types
	cdef double p43_x, p43_y, p21_x, p21_y, u, a, b, x, y

	p43_x = p3_x - p4_x
	p43_y = p3_y - p4_y
	p21_x = p1_x - p2_x
	p21_y = p1_y - p2_y

	u = p21_x * p43_y - p21_y * p43_x
	
	if u > 0:
		a = p3_x*p4_y - p3_y*p4_x
		b = p1_x*p2_y - p1_y*p2_x

		x = (b * p43_x - p21_x * a) / u
		y = (b * p43_y - p21_y * a) / u

		# Check if on second line segment or outside
		# & give it some elasticity
		if double_min(p3_x, p4_x) * el_1 <= x <= double_max(p3_x, p4_x) * el_2 and double_min(p3_y, p4_y) * el_1 <= y <= double_max(p3_y, p4_y) * el_2:
			return x, y

	# Tuple return type...
	return False, False


cpdef double distance(double p1_x, double p1_y, double p2_x, double p2_y, ):
	cdef double v_x, v_y
	v_x = p2_x - p1_x
	v_y = p2_y - p1_y
	return c_libc_sqrt((v_x * v_x) + (v_y * v_y))


cpdef np.ndarray[double, ndim=1] find_max_dist(list barriers, double cp_x, double cp_y, double max_len_spike, list angles, int ignore_start, int  ignore_end):
	
	#Type declarations
	cdef double o_x, o_y, len_inter, dist, p1_x, p1_y, p2_x, p2_y
	cdef int i, j, num_barrier
	cdef tuple p_inter
	cdef list b
	cdef np.ndarray[double, ndim=1] max_dist

	num_barriers = len(barriers)

	max_dist = numpy.empty((len(angles)))

	cdef double max_x, min_x, max_y, min_y
	max_x = cp_x + max_len_spike
	min_x = cp_x - max_len_spike
	max_y = cp_y + max_len_spike
	min_y = cp_y - max_len_spike

	ind_list = range(ignore_start) + range(ignore_end, len(barriers))

	for i in range(len(angles)):
		o_x = max_len_spike * c_libc_cos(angles[i]) + cp_x
		o_y = max_len_spike * c_libc_sin(angles[i]) + cp_y

		len_inter = max_len_spike

		for j in ind_list:
			b = barriers[j]
			# Check if one of the points lies within max_len_spike (square)
			if ((min_x < b[0] < max_x) and (min_y < b[1] < max_y)) or ((min_x < b[2] < max_x) and (min_y < b[3] < max_y)):
				#Check on which side of the line the point is & call cython optimized function
				if ((b[2] - b[0])*(cp_y - b[1]) - (b[3] - b[1])*(cp_x - b[0])) > 0:
					p_inter = intersect_lines(cp_x, cp_y, o_x, o_y, b[0], b[1], b[2], b[3])
				else:
					p_inter = intersect_lines(o_x, o_y, cp_x, cp_y, b[0], b[1], b[2], b[3])

				if p_inter != (False, False):
					dist = distance(p_inter[0], p_inter[1], cp_x, cp_y)
					if dist < len_inter:
						len_inter = dist

		max_dist[i] = len_inter

	return max_dist

# Find max dist multiple (simultaniously)
cpdef find_shape_multiple(np.ndarray[double, ndim=2] center_points, np.ndarray[double, ndim=1] angles, 
		np.ndarray[double, ndim=3] barriers, double max_len):

	num_droplets = center_points.shape[0]	
	spikes_per_droplet = angles.shape[0]

	spikes = numpy.array([[0.0 for i in range(spikes_per_droplet)] for j in range(num_droplets)])
	running = numpy.array([[True for i in range(spikes_per_droplet)] for j in range(num_droplets)])
	found = False

	cdef double max_x, min_x, max_y, min_y
	found = 0

	while found < 10:
		for this in numpy.ndindex(spikes.shape):
			if running[this]:
				spikes[this] += 0.1
				cp_x = center_points[this[0]][0]
				cp_y = center_points[this[0]][1]

				max_x = cp_x + max_len
				min_x = cp_x - max_len
				max_y = cp_y + max_len
				min_y = cp_y - max_len

				o_x = spikes[this] * c_libc_cos(angles[this[1]]) + cp_x
				o_y = spikes[this] * c_libc_sin(angles[this[1]]) + cp_y

				# Check barriers
				for j in range(barriers.shape[0]):
					b = barriers[j]
					# Check if one of the points lies within max_len_spike (square)
					if ((min_x < b[0][0] < max_x) and (min_y < b[0][1] < max_y)) or ((min_x < b[1][0] < max_x) and (min_y < b[1][1] < max_y)):
						#Check on which side of the line the point is & call cython optimized function
						if ((b[1][0] - b[0][0])*(cp_y - b[0][1]) - (b[1][1] - b[0][1])*(cp_x - b[0][0])) > 0:
							p_inter = intersect_lines(cp_x, cp_y, o_x, o_y, b[0][0], b[0][1], b[1][0], b[1][1])
						else:
							p_inter = intersect_lines(o_x, o_y, cp_x, cp_y, b[0][0], b[0][1], b[1][0], b[1][1])

						if p_inter != (False, False):
							running[this] = False	

				# Check other spikes 
				if running[this]:
					for other in numpy.ndindex(spikes.shape):
						if other[0] != this[0]:
							p1_x = center_points[other[0]][0]
							p1_y = center_points[other[0]][1]
							p2_x = spikes[other] * c_libc_cos(angles[other[1]]) + p1_x
							p2_y = spikes[other] * c_libc_sin(angles[other[1]]) + p1_y

							if ((min_x < p1_x < max_x) and (min_y < p1_y < max_y)) or ((min_x < p2_x < max_x) and (min_y < p2_y < max_y)):
								#Check on which side of the line the point is & call cython optimized function
								if ((p2_x - p1_x)*(cp_y - p1_y) - (p2_y - p1_y)*(cp_x - p1_x)) > 0:
									p_inter = intersect_lines(cp_x, cp_y, o_x, o_y, p1_x, p1_y, p2_x, p2_y)
								else:
									p_inter = intersect_lines(o_x, o_y, cp_x, cp_y, p1_x, p1_y, p2_x, p2_y)

								if p_inter != (False, False):
									running[this] = False
									running[other] = False

		found += 1

	return spikes

cpdef get_area(spikes, mult_term):
	for i in range(len(spikes)):
		i_next = ()
	return np.sum(self._spikes * np.roll(self._spikes, 1) * self._mult_term)

cpdef np.ndarray[long, ndim=1] downsample(np.ndarray[double, ndim=2] points, double rate):
	# Returns a list with indices of all used points
	cdef int i, num_points, before, after
	cdef list ind
	cdef tuple j
	cdef np.ndarray[long, ndim=1] res

	num_points = len(points)
	ind = []

	for i in range(num_points):
		if i%5:
			before = (i-1)%num_points
			after = (i+1)%num_points
			ind.append((distance(points[before, 0], points[before, 1], points[after, 0], points[after, 1]), i))
		else:
			ind.append((10000000, i))

	ind.sort(reverse = True)
	res = numpy.array(sorted([j[1] for j in ind[:int(rate * num_points)]]))
	return res





