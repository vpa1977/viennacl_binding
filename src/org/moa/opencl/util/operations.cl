
// #define VALUE_TYPE double
enum   AttributeType
{
		atNUMERIC,
		atNOMINAL
};

__kernel void double2uint(__global double* in, __global  uint* out, const uint scale)
{
	int id =get_global_id(0);
	out[id] = (uint)(in[id] * scale);
}


__kernel void normalize_attributes(
		__global double* in,
		__global double* out,
		__global  double* range_min,
		__global  double* range_max,
		__global const int* attribute_type,
		const int attribute_size)
{
	int vector_offset = attribute_size * get_global_id(0);
	int i;
	for (i = 0; i < attribute_size ; i ++ )
	{
		if (attribute_type[i] == atNUMERIC)
		{
			double width = ( range_max[i] - range_min[i]);
			out[vector_offset+i] = width > 0 ? (in[vector_offset+i] - range_min[i]) / width : 0;
		}
		else
		{
			out[vector_offset+i] = in[vector_offset + i];
		}
	}


}
