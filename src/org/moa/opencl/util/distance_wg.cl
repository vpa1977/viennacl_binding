

//#define VALUE_TYPE double
//enum   AttributeType
//{
//		atNUMERIC,
//		atNOMINAL
//};

inline void workgroup_reduce(__local VALUE_TYPE* data)
{
	int lid = get_local_id(0);
	int size = get_local_size(0);
	for (int d = size >>1 ;d > 0 ; d >>=1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
	 	if (lid < d)
	 	{
	 		data[lid] += data[lid + d];
	 	}
	 }
	 barrier(CLK_LOCAL_MEM_FENCE);
}

/*
   distance kernel two - 1 instance per workgroup
*/
__kernel void square_distance_one_wg(__global const VALUE_TYPE* input,
						__global const VALUE_TYPE* samples,
						__global const VALUE_TYPE* range_min,
						__global const VALUE_TYPE* range_max,
						__global const int* attribute_type,
						__global VALUE_TYPE* result,
						const int length,
						const int attribute_size,
						__local VALUE_TYPE* scratch)
{
	VALUE_TYPE accumulator = 0;
	int vector_offset = get_group_id(0) * attribute_size;
	for (int attribute_id = get_local_id(0); attribute_id < attribute_size; attribute_id += get_local_size(0) )
	{

		if (attribute_type[attribute_id] == atNUMERIC)
		{
			VALUE_TYPE width = ( range_max[attribute_id] - range_min[attribute_id]);
			VALUE_TYPE val = width > 0 ? (input[attribute_id] - range_min[attribute_id]) / width  - (samples[ vector_offset + attribute_id] - range_min[attribute_id])/width : 0;
			accumulator += val*val;
		}
		else
		{
			accumulator += isnotequal( input[attribute_id] , samples[vector_offset + attribute_id]);
		}
	}
	scratch[get_local_id(0)] = accumulator;
	workgroup_reduce(scratch);
	result[get_group_id(0)] = scratch[0];
}



