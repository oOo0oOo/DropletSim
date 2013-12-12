# Inline min and max functions
cdef inline float float_max(float a, float b): return a if a >= b else b
cdef inline float float_min(float a, float b): return a if a <= b else b

# sqrt, cos and sin
cdef extern from "stdlib.h":
	double c_libc_sqrt "sqrt"(double x)

cdef extern from "stdlib.h":
	double c_libc_cos "cos"(double x)

cdef extern from "stdlib.h":
	double c_libc_sin "sin"(double x)


#Elasticity parameters
cdef float el_1, el_2
el_1 = 0.999
el_2 = 2-el_1

cpdef tuple intersect_lines(float p1_x, float p1_y, float p2_x, float p2_y, float p3_x, float p3_y, float p4_x, float p4_y):
	# Declare all the types
	cdef float p43_x, p43_y, p21_x, p21_y, u, a, b, x, y

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
		# Not very versatile...
		# give it some elasticity
		if float_min(p3_x, p4_x) * el_1 <= x <= float_max(p3_x, p4_x) * el_2 and float_min(p3_y, p4_y) * el_1 <= y <= float_max(p3_y, p4_y) * el_2:
			return x, y

	return False, False


cpdef double distance(float p1_x, float p1_y, float p2_x, float p2_y, ):
	cdef double v_x, v_y
	v_x = p2_x - p1_x
	v_y = p2_y - p1_y
	return c_libc_sqrt((v_x * v_x) + (v_y * v_y))


cpdef list find_max_dist(list barriers, float cp_x, float cp_y, float max_len_spike, list angles):
	
	#Type declarations
	cdef float o_x, o_y, len_inter, dist, p1_x, p1_y, p2_x, p2_y
	cdef int i
	cdef list max_dist = angles[:]
	cdef tuple p_inter

	for i in range(len(angles)):
		o_x = max_len_spike * c_libc_cos(angles[i]) + cp_x
		o_y = max_len_spike * c_libc_sin(angles[i]) + cp_y

		len_inter = max_len_spike

		for p1_x, p1_y, p2_x, p2_y in barriers:

			#Check on which side of the line the point is & call cython optimized function
			if ((p2_x - p1_x)*(cp_y - p1_y) - (p2_y - p1_y)*(cp_x - p1_x)) > 0:
				p_inter = intersect_lines(cp_x, cp_y, o_x, o_y, p1_x, p1_y, p2_x, p2_y)
			else:
				p_inter = intersect_lines(o_x, o_y, cp_x, cp_y, p1_x, p1_y, p2_x, p2_y)

			if p_inter != (False, False):
				dist = distance(p_inter[0], p_inter[1], cp_x, cp_y)
				if dist < len_inter:
					len_inter = dist

		max_dist[i] = len_inter

	return max_dist