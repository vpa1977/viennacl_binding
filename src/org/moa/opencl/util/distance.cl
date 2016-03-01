

// #define VALUE_TYPE double
enum   AttributeType
{
		atNUMERIC,
		atNOMINAL
};





/*
   distance kernel one
*/
void square_distance_small(
            const int N, 
            __global const VALUE_TYPE* input,
						__global const VALUE_TYPE* samples,
						__global const VALUE_TYPE* range_min,
						__global const VALUE_TYPE* range_max,
						__global const int* attribute_type,
						__global const int* indices,
						__global VALUE_TYPE* result,
						const int attribute_size,
						const int use_indices, const int indices_offset)
{
  for (int id = get_global_id(0) ; id < N ; id += get_global_size(0)) 
  {
    int result_offset =  use_indices == 1 ? indices[get_global_id(0)+indices_offset] : get_global_id(0);

    int vector_offset = attribute_size * result_offset;
    VALUE_TYPE point_distance = 0;
    VALUE_TYPE val;
    VALUE_TYPE width;
    int i;
    for (i = 0; i < attribute_size ; i ++ )
    {
      int s = samples[vector_offset + i];
      int v = input[i];
      width = ( range_max[i] - range_min[i]);
      val = select((VALUE_TYPE)0.0, (VALUE_TYPE)(input[i] - range_min[i]) / width  - (samples[ vector_offset + i] - range_min[i])/width ,(COND_TYPE)( width > 0));
      VALUE_TYPE nominal = select((VALUE_TYPE)0.0,(VALUE_TYPE)1.0,(COND_TYPE)(s!=v));
      point_distance += select((VALUE_TYPE) nominal,(VALUE_TYPE) val * val, (COND_TYPE)(attribute_type[i] == atNUMERIC));

    }
    result[get_global_id(0)] = point_distance;
  }
}

void square_distance_large(
			__local VALUE_TYPE* cache,
            const int N,
            __global const VALUE_TYPE* input,
						__global const VALUE_TYPE* samples,
						__global const VALUE_TYPE* range_min,
						__global const VALUE_TYPE* range_max,
						__global const int* attribute_type,
						__global const int* indices,
						__global VALUE_TYPE* result,
						const int attribute_size,
						const int use_indices, const int indices_offset)
{

  int lid = get_local_id(0);
  for (int id = get_group_id(0) ; id < N ; id += get_num_groups(0)) // each group processes 1 instance then jumps to the next one.
  {
    int result_offset =  use_indices == 1 ? indices[id+indices_offset] : id;

    int vector_offset = attribute_size * result_offset;
    VALUE_TYPE point_distance = 0;
    VALUE_TYPE val;
    VALUE_TYPE width;
    int i =lid;
    // compute thread local distance
    for (; i < attribute_size ; i +=get_local_size(0))
    {
      int s = samples[vector_offset + i];
      int v = input[i];
      width = ( range_max[i] - range_min[i]);
      val = select((VALUE_TYPE)0.0, (VALUE_TYPE)(input[i] - range_min[i]) / width  - (samples[ vector_offset + i] - range_min[i])/width ,(COND_TYPE)( width > 0));
      VALUE_TYPE nominal = select((VALUE_TYPE)0.0,(VALUE_TYPE)1.0,(COND_TYPE)(s!=v));
      point_distance += select((VALUE_TYPE) nominal,(VALUE_TYPE) val * val, (COND_TYPE)(attribute_type[i] == atNUMERIC));
    }
    cache[lid] = point_distance;
	#pragma unroll 8
    for (int d = get_local_size(0) >>1 ;d > 0 ; d >>=1)
	{
   		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < d)
		{
			cache[lid] += cache[lid+d];
		}
     }
	barrier(CLK_LOCAL_MEM_FENCE);
	if (lid ==0)
		result[id] =   cache[0];
  }
}

__kernel void square_distance(
            const int N,
            __global const VALUE_TYPE* input,
						__global const VALUE_TYPE* samples,
						__global const VALUE_TYPE* range_min,
						__global const VALUE_TYPE* range_max,
						__global const int* attribute_type,
						__global const int* indices,
						__global VALUE_TYPE* result,
						const int attribute_size,
						const int use_indices, const int indices_offset)
{
	__local VALUE_TYPE cache[256];
	if (attribute_size >= 256)
		square_distance_large(cache,N, input, samples, range_min, range_max, attribute_type, indices, result, attribute_size, use_indices, indices_offset);
	else
		square_distance_small(N, input, samples, range_min, range_max, attribute_type, indices, result, attribute_size, use_indices, indices_offset);
}
