import c_code
import random

data = [random.randrange(100) for i in range(8)]
for i in xrange(1000000):
	c_code.intersect_lines(*data)