
// #define VALUE_TYPE double
enum   AttributeType
{
		atNUMERIC,
		atNOMINAL
};

__kernel void double2uint(__global double* in, __global uint* attribute_map, __global  uint* out, const uint scale, const uint max, const uint attribute_size)
{
  int id =get_global_id(0);

  double value = in[id];
  if (id < max)
  {
	  if (attribute_map[id % attribute_size ] == atNUMERIC)
		  out[id] = (uint)(value* scale);
	  else
		  out[id] = ((uint)in[id]) * scale;
  }

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

