
/*
   distance kernel two
*/
__kernel void cosine_distance(__global const VALUE_TYPE* input,
						__global const VALUE_TYPE* samples,
						__global const int* attribute_type,
						__global const int* indices,
						__global VALUE_TYPE* result,
						const int attribute_size,
						const int use_indices)
{

	int result_offset =  use_indices == 1 ? indices[get_global_id(0)] : get_global_id(0);

	int vector_offset = attribute_size * result_offset;
	VALUE_TYPE point_distance = 0;
	VALUE_TYPE val= 0;
	VALUE_TYPE width;
	VALUE_TYPE norm_src = 0;
	VALUE_TYPE norm_sample = 0;
	int i;
	for (i = 0; i < attribute_size ; i ++ )
	{
		int s = samples[vector_offset + i];
		int v = input[i];
		double in = input[i];
		double sample = samples[vector_offset +i];
		val = in * sample;
		double nominal = select(0.0,1.0,(ulong)(s!=v));
		point_distance += select( nominal, val, (ulong)(attribute_type[i] == atNUMERIC));
		norm_src += select( 1.0, in*in , (ulong)(attribute_type[i] == atNUMERIC));
		norm_sample += select( 1.0, sample*sample , (ulong)(attribute_type[i] == atNUMERIC));

	}
	VALUE_TYPE divisor = norm_src * norm_sample;
	result[get_global_id(0)] = 1 - acospi(select(0.0, point_distance/divisor, (ulong)(divisor != 0)));
}


/*
   distance kernel three
*/
__kernel void cosine_distance_norm(__global const VALUE_TYPE* input,
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
	VALUE_TYPE val= 0;
	VALUE_TYPE width;
	VALUE_TYPE norm_src = 0;
	VALUE_TYPE norm_sample = 0;
	int i;
	for (i = 0; i < attribute_size ; i ++ )
	{
		int s = samples[vector_offset + i];
		int v = input[i];
		double in = input[i];
		double sample = samples[vector_offset +i];
		width = ( range_max[i] - range_min[i]);
		val = select(0.0, ((in - range_min[i]) / width)* ((sample - range_min[i])/width) ,(ulong)( width > 0));
		double nominal = select(0.0,1.0,(ulong)(s!=v));
		point_distance += select( nominal, val, (ulong)(attribute_type[i] == atNUMERIC));
		norm_src += select( 1.0, in*in , (ulong)(attribute_type[i] == atNUMERIC));
		norm_sample += select( 1.0, sample*sample , (ulong)(attribute_type[i] == atNUMERIC));

	}
	VALUE_TYPE divisor = norm_src * norm_sample;
	result[get_global_id(0)] = 2 - select(0.0, point_distance/divisor, (ulong)(divisor != 0));
}


