__kernel void simple_update( const int columns,
		 const int rows,
		const double learning_rate,
		__global const double *gradients,
		__global double* weights)
{
	int base_thread_id = get_group_id(0) * get_local_size(0) + get_local_id(0)
	for (int i = base_thread_id ; i < columns ; i+= get_local_size(0))
	{
		double update_value = learning_rate * gradients[i] / rows;
		old = weights[i];
		sum = old + f;
		atom_cmpxchg((volatile __global ulong*)(&weights[i]),as_ulong(old), as_ulong(sum));

	}

}
