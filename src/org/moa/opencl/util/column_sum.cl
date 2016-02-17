// local_size(256,1)
// global_size
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#define WAVEFRONT_SIZE 64


__kernel void column_sum_2
(
		__global const unsigned int * column_indices,
		const unsigned int num_elements,
		const unsigned int num_columns,
		__global VALUE_TYPE * elements,
		__global VALUE_TYPE * reduction_result
)
{

	__local VALUE_TYPE column_sum[1024];
	int group_id = get_group_id(0);
	int start_element = group_id * get_local_size(0) + get_local_id(0);
	int stride = get_num_groups(0) * get_local_size(0);
	int base_column = get_group_id(1) * 1024;

	for (int base_column = 0 ; base_column < num_columns ; base_column += 1024)
	{

		for (int j = get_local_id(0); j < 1024; j += get_local_size(0))
			column_sum[j] = 0;

		for (int j = start_element; j < num_elements; j+=stride) // parallel wg then jump
		{
				int column = column_indices[j];
				VALUE_TYPE update_value = elements[j];

				int idx = column- base_column;
				int cond = (idx < 1024) && (idx >=0) && update_value > 0;
				if (cond)
				{
					double old, sum;
					volatile ulong test;
					do
					{
						old = column_sum[idx];
						sum = old + update_value;
					} while ( (test = atom_cmpxchg((volatile __local ulong*)(&column_sum[idx]),as_ulong(old), as_ulong(sum))) != as_ulong(old));
				}

		}


		unsigned int limit = min((unsigned int)1024, (unsigned int)(num_columns - base_column));

		for (unsigned int j = get_local_id(0); j < limit; j += get_local_size(0))
		{
			double f = column_sum[j];
			double old, sum;
			volatile ulong test;
			do
			{
				old = reduction_result[base_column+j];
				sum = old + f;
			}  while ( (test = atom_cmpxchg((volatile __global ulong*)(&reduction_result[base_column+j]),as_ulong(old), as_ulong(sum))) != as_ulong(old));

		}

	}

}

