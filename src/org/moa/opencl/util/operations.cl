
// #define VALUE_TYPE double
enum   AttributeType
{
		atNUMERIC,
		atNOMINAL
};

__kernel void double2uint(__global double* in, __global uint* attribute_map, __global  uint* out, const uint scale, const uint max, const uint attribute_size)
{
  int id =get_global_id(0);
  for (; id < max ; id+= get_global_size(0))
  {
	  double value = in[id];
	  if (attribute_map[id % attribute_size ] == atNUMERIC)
		  out[id] = (uint)(value* scale);
	  else
		  out[id] = ((uint)in[id]) * scale;
  }

}

__kernel void prepare_order_key(__global uint* buffer, int max)
{

  int id = get_global_id(0);
  for (; id < max; id += get_global_size(0))
	  buffer[id] = id;
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
	 for (int j = get_global_id(0); j < num_instances*attribute_size; j+= get_global_size(0))
	 {
		 int attribute = j % attribute_size;
    	if (attribute_type[attribute] == atNUMERIC)
		{
			double width = ( range_max[attribute] - range_min[attribute]);
			out[j] = width > 0 ? (in[j] - range_min[attribute]) / width : 0;
		}
		else
		{
			out[j] = in[j];
		}
	 }
}

__kernel void normalize_attributes_float(
		__global float* in,
		__global float* out,
		__global  float* range_min,
		__global  float* range_max,
		__global const int* attribute_type,
		const int attribute_size,
    const int num_instances)
{
  for (int j = get_global_id(0); j < num_instances; j+= get_global_size(0))
  {
	int vector_offset = attribute_size * j;
	int i;
	for (i = 0; i < attribute_size ; i ++ )
	{
		if (attribute_type == 0)
		{
			double width = ( range_max[i] - range_min[i]);
			out[vector_offset+i] = width > 0 ? (in[vector_offset+i] - range_min[i]) / width : 0;
		}
		else if (attribute_type[i] == atNUMERIC)
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
}

__kernel void normalize_attributes_float_replace_min(
		__global float* in,
		__global float* out,
		__global  float* range_min,
		__global  float* range_max,
		__global const int* attribute_type,
		const int attribute_size,
    const int num_instances)
{
  for (int j = get_global_id(0); j < num_instances; j+= get_global_size(0))
  {
	int vector_offset = attribute_size * j;
	int i;
	for (i = 0; i < attribute_size ; i ++ )
	{
		if (attribute_type == 0)
		{
			float width = ( range_max[i] - range_min[i]);
			out[vector_offset+i] = width > 0 ? (in[vector_offset+i] - range_min[i]) / width : 0;
			out[vector_offset +i] = out[vector_offset +i]> 0 ? out[vector_offset +i] : 0;
		}
		else if (attribute_type[i] == atNUMERIC)
		{
			float width = ( range_max[i] - range_min[i]);
			out[vector_offset+i] = width > 0 ? (in[vector_offset+i] - range_min[i]) / width : 0;
			out[vector_offset +i] = out[vector_offset +i]> 0 ? out[vector_offset +i] : 0;
		}
		else
		{
			out[vector_offset+i] = in[vector_offset + i];
		}
	}
  }
}



__kernel void random_shift(__global uint* data, __global uint* shift, const int max)
{
	int attrib_id = get_global_id(0);
	int instance_id = get_global_id(1);
  for (;instance_id < max; instance_id+= get_global_size(1))
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

