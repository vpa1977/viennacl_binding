#define SQRT_2 1.41421356237
// 0 -> number of instances
// 1 -> number of attributes
__kernel void permute(const int k,
					  __global const  DATA_TYPE * src,
					  __global const int* perm,
					  DATA_TYPE srhst_const,
					  __global  DATA_TYPE * out)
{
	for (int i = get_global_id(0); i < k; i+= get_global_size(0))
		out[i] = SQRT_2 * srhst_const * src[ perm[i]];
};

