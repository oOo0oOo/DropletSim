c_source = r'''
__global__ void check_stop(int *stopped, float *x_coords, float *y_coords, float *center_x, float *center_y, int per_droplet, int total_number)
{
	const int i = blockDim.x*blockIdx.x + threadIdx.x;
	const float threshold = 3.5;
	const int max_dist = 200;

	if (stopped[i] == 0){
		const int num = (i - (i % per_droplet)) / per_droplet;
		const float c_x = center_x[num];
		const float c_y = center_y[num];

		float distance, min_dist, v_x, v_y, x_i, y_i;
		int start, stop;

		x_i = x_coords[i];
		y_i = y_coords[i];
		min_dist = 1000000.0;

		for(int m = 0; m < num; m++) {
			if (abs(center_x[m] - c_x) < max_dist && abs(center_y[m] - c_y) < max_dist){
				start = m * per_droplet;
				stop = (m + 1) * per_droplet;
				for(int n = start; n < stop; n++) {
					v_x = x_coords[n] - x_i;
					v_y = y_coords[n] - y_i;

					distance = sqrt(v_x * v_x + v_y * v_y);
					if (distance < min_dist){
						min_dist = distance;
					};
				};
			};
		
		};

		for(int m = num + 1; m < total_number / per_droplet; m++) {
			if (abs(center_x[m] - c_x) < max_dist && abs(center_y[m] - c_y) < max_dist){
				start = m * per_droplet;
				stop = (m + 1) * per_droplet;

				for(int n = start; n < stop; n++) {
					v_x = x_coords[n] - x_i;
					v_y = y_coords[n] - y_i;

					distance = sqrt(v_x * v_x + v_y * v_y);
					if (distance < min_dist){
						min_dist = distance;
					};
				};
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



c_source_naive = r'''
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