

// #define VALUE_TYPE double
enum   AttributeType
{
		atNUMERIC,
		atNOMINAL
};




/*
   distance kernel one
*/
__kernel void square_distance(__global const VALUE_TYPE* input,
						__global const VALUE_TYPE* samples,
						__global const VALUE_TYPE* range_min,
						__global const VALUE_TYPE* range_max,
						__global const int* attribute_type,
						__global const int* indices,
						__global VALUE_TYPE* result,
						const int attribute_size,
						const int use_indices)
{

	int result_offset =  use_indices == 1 ? indices[get_global_id(0)] : get_global_id(0);

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
		val = select(0.0, (input[i] - range_min[i]) / width  - (samples[ vector_offset + i] - range_min[i])/width ,(ulong)( width > 0));
		double nominal = select(0.0,1.0,(ulong)(s!=v));
		point_distance += select( nominal, val * val, (ulong)(attribute_type[i] == atNUMERIC));

	}
	result[get_global_id(0)] = point_distance;
}


