__kernel void piCal(
    __global float *pi_out)
    
{
	//Select amount of fraction sets per work item
	int num_frac = 4;
	//Offset between fraction set
	int frac_space = 4;
	int num_loop;
	int div;
	float num = 1.0f;

	//Get global work item ID
	int item = get_global_id (0);

	//Starting offset of fraction set
	int term = item * num_frac * frac_space;

	//Loop for all fractions inside a fraction set
	for (num_loop = 0; num_loop < num_frac; num_loop++){
	
		div = num_loop * frac_space + term;
	
		pi_out[item] += (num / (div + 1)) - (num / (div + 3));
	}
}
