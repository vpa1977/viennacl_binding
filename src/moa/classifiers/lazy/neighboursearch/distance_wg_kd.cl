

#define VALUE_TYPE double
enum   AttributeType
{
		atNUMERIC,
		atNOMINAL
};

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

__kernel void square_distance_index(
            __global const int* indices,
            __global const VALUE_TYPE* input,
						__global const VALUE_TYPE* samples,
						__global const VALUE_TYPE* range_min,
						__global const VALUE_TYPE* range_width,
						__global const int* attribute_type,
						__global VALUE_TYPE* result,
						const int attribute_size,
            const int start
						)
{

	int result_offset = indices[get_global_id(0) + start];

	int vector_offset = attribute_size * result_offset;
	VALUE_TYPE point_distance = 0;
	VALUE_TYPE val;
	VALUE_TYPE width;
	int i;
	for (i = 0; i < attribute_size ; i ++ )
	{
		if (attribute_type[i] == atNUMERIC)
		{
			width = range_width[i];
			val = width > 0 ? (input[i] - range_min[i]) / width  - (samples[ vector_offset + i] - range_min[i])/width : 0;
			point_distance += val*val;
		}
		else
		{
			point_distance += isnotequal( input[i] , samples[vector_offset + i]);
		}
	}
	result[get_global_id(0)] = point_distance;
}



/*
   distance kernel two - 1 instance per workgroup
*/
__kernel void square_distance_one_wg(
						__global const int* sample_indices,
						__global const VALUE_TYPE* input,
						__global const VALUE_TYPE* samples,
						__global const VALUE_TYPE* range_min,
						__global const VALUE_TYPE* range_width,
						__global const int* attribute_type,
						__global VALUE_TYPE* result,
						const int attribute_size,
						const int offset
						)
{
	__local VALUE_TYPE scratch[256];
	VALUE_TYPE accumulator = 0;
	int vector_offset = sample_indices[offset + get_group_id(0)] * attribute_size;
	for (int attribute_id = get_local_id(0); attribute_id < attribute_size; attribute_id += get_local_size(0) )
	{

		if (attribute_type[attribute_id] == atNUMERIC)
		{
			VALUE_TYPE width = range_width[attribute_id];
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



