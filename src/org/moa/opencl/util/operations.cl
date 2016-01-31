
// #define VALUE_TYPE double
enum   AttributeType
{
		atNUMERIC,
		atNOMINAL
};

__kernel void double2uint(__global double* in, __global  uint* out, const uint scale, const uint max)
{
	int id =get_global_id(0);
  if (id < max)
    out[id] = (uint)(in[id] * scale);
}

__kernel void prepare_order_key(__global uint* buffer, int max)
{
  int id = get_global_id(0);
  if (id < max) buffer[id] = id;
}


__kernel void normalize_attributes(
		__global double* in,
		__global double* out,
		__global  double* range_min,
		__global  double* range_max,
		__global const int* attribute_type,
		const int attribute_size,
    const int num_instances)
{
  if (get_global_id(0) >= num_instances) 
    return;
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

__kernel void random_shift(__global uint* data, __global uint* shift, const int max)
{
	int attrib_id = get_global_id(0);
	int instance_id = get_global_id(1);
  if (instance_id < max) 
  {
    int instance_size = get_global_size(0);
    int offset = instance_id * instance_size + attrib_id;
    data[offset]+= shift[attrib_id];
  }

}

enum CompareFlags
{
	enNONE,
	enGREATER,
	enLESS
};

/*
__kernel void binary_search(        __global uint4 * outputArray,
		__global const uint  * sortedArray, // sorted code indices
		__global const  uchar* morton_codes, // computed morton code block
		__global const  uchar* find_me,
		const   unsigned int code_len,
        const   unsigned int globalLowerBound,
        const   unsigned int globalUpperBound,
        const   unsigned int subdivSize)
{
	unsigned int tid = get_global_id(0);

	unsigned int lowerBound = globalLowerBound + subdivSize * tid;
	unsigned int upperBound = lowerBound + subdivSize - 1;

	unsigned int lowerBoundElement = sortedArray[lowerBound];
	unsigned int upperBoundElement = sortedArray[upperBound];

	unsigned int lowerOffset = lowerBoundElement * code_len;
	unsigned int upperOffset = upperBoundElement * code_len;
	CompareFlags lowFlag = enNONE;
	CompareFlags highFlag = enNONE;
	for (int i = 0; i < code_len; ++i)
	{
		if (lowFlag == enNONE)
		{
			if (morton_codes[lowerOffset +i] < find_me[i])
				lowFlag = enLESS;
			if (morton_codes[lowerOffset +i] > find_me[i])
				lowFlag = enGREATER;

			if (morton_codes[upperOffset +i] < find_me[i])
				highFlag = enLESS;
			if (morton_codes[upperOffset +i] > find_me[i])
				highFlag = enGREATER;
		}

	}

	if( (highFlag == enGREATER) || (lowFlag == enLESS))
	{
	   return;
	}
	else
	{
	   outputArray[0].x = lowerBound;
	   outputArray[0].y = upperBound;
	   outputArray[0].w = 1;
	}
}
*/
