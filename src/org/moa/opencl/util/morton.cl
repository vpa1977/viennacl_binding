// global_size(1) == 3
__kernel void morton_code_backup(__global uchar* result,
													__global uchar* lookup_table,
													__global uint* points,
													const uint dimensions,
													const uint N)
{
		const uint BYTE_COUNT = 256;
		const uint rightshift = 32;
		const int total_len = dimensions * 32 / 8;
		uint size_one = get_global_size(1);
		uint i_depth = get_global_id(1) +1;
	  for (uint id = get_global_id(0); id < N ; id+= get_global_size(0))
		{
			uint input_offset =  id * dimensions;
			uint output_offset = id *total_len;

			//for (uint i_depth = 1; i_depth <= 4; ++i_depth)
			{
					int result_offset = 3 * dimensions - (i_depth - 1)*dimensions; // each mask has [dimensions] bytes in it
					for (int pos = result_offset; pos < result_offset + dimensions; ++pos)
					{
						for (unsigned int d = 0; d < dimensions; ++d)
						{
							uint point_value = points[input_offset + d];
							uint shifted_value =  point_value >> (rightshift - 8 * i_depth);
							uint byte = shifted_value & 0xFF;
							uint offset = d* dimensions * BYTE_COUNT + byte * dimensions + pos - result_offset;
							result[output_offset + pos] |= lookup_table[offset];
						}
					}
				}

		}
}



// global_size(1) == 3
__kernel void morton_code(__global uchar* result,
													__global uchar* lookup_table,
													__global uint* points,
													const uint dimensions,
													const uint N)
{
		const uint BYTE_COUNT = 256;
		const uint rightshift = 32;
		const int total_len = dimensions * 32 / 8;

		const int long_steps = total_len / 8;
	    for (uint id = get_global_id(0); id < N ; id+= get_global_size(0))
		{
			uint input_offset =  id * dimensions;
			uint output_offset = id *total_len;
			//int i_depth = get_global_id(1)+1;
			for (int i_depth = 1; i_depth <= 4; ++i_depth)
			{
				int result_offset = output_offset + 3 * dimensions - (i_depth - 1)*dimensions; // each mask has [dimensions] bytes in it
				int depth = (rightshift - 8 * i_depth);
				for (unsigned int d = 0; d < dimensions; ++d)
				{
					uint point_value = points[input_offset + d];
					uint byte =  (point_value >> depth) & 0xFF;
					uint offset = d* dimensions * BYTE_COUNT + byte * dimensions;
					uint byte_pos = long_steps * 8;
					__global long* result_as_long = (__global long*)&result[result_offset];
					__global long* lookup_as_long = (__global long*)&lookup_table[offset ];
					for (int pos =0 ; pos < long_steps; ++pos)
					{
						result_as_long[pos] |= lookup_as_long[pos];
					}
					for (int pos =byte_pos ; pos < dimensions; ++pos)
					{
						result[result_offset + pos] |= lookup_table[offset+ pos ];
					}
				}
			}

		}
}


__kernel void morton_code_group(__global uchar* result,
													const __global uchar* lookup_table,
													__global const uint* points,
													const uint dimensions,
													const uint N)
{
		const uint BYTE_COUNT = 256;
		const uint rightshift = 32;
		const int total_len = dimensions * 32 / 8;
		const int long_steps = dimensions/8;
		const uint byte_pos = long_steps * 8;
		__global long* result_as_long = (__global long*)&result[0];
		__global long* lookup_as_long = (__global long*)&lookup_table[0];

		__global uint* result_as_uint = (__global uint*)&result[0];
		__global uint* lookup_as_uint = (__global uint*)&lookup_table[0];
		__local ulong buffer[256];
		__local uint point_dim[1024];
	    for (uint id = get_group_id(0); id < N ; id+= get_num_groups(0))
		{
			uint input_offset =  id * dimensions;
			uint output_offset = id *total_len;

			for (int j = get_local_id(0); j < dimensions; j+= get_local_size(0))
				point_dim[j] = points[input_offset + j];

			//int i_depth = get_global_id(1)+1;
			for (int i_depth = 1; i_depth <= 4; ++i_depth)
			{
				int result_offset = output_offset + 3 * dimensions - (i_depth - 1)*dimensions; // each mask has [dimensions] bytes in it
				int depth = (rightshift - 8 * i_depth);
				for (unsigned int d = 0; d < dimensions; ++d)
				{
					uint point_value = point_dim[d]; //points[input_offset + d]
					uint byte =  (point_value >> depth) & 0xFF;
					uint offset = d* dimensions * BYTE_COUNT + byte * dimensions;
					uint lr_offset = result_offset/8;
					uint l_offset = offset/8;
					for (int pos =get_local_id(0) ; pos < long_steps; pos += get_local_size(0))
					{
						result_as_long[lr_offset + pos] |= lookup_as_long[l_offset+ pos];
					}
					barrier(CLK_LOCAL_MEM_FENCE);

					for (int pos =byte_pos+ get_local_id(0) ; pos < dimensions;  pos += get_local_size(0))
					{
						result[result_offset + pos] |= lookup_table[offset+ pos ];
					}
					barrier(CLK_LOCAL_MEM_FENCE);
				}
			}

		}
}

// workgroup size 64
__kernel void morton_code_group2(__global uchar* result,
													const __global uchar* lookup_table,
													__global const uint* points,
													const uint dimensions,
													const uint N)
{
		const uint BYTE_COUNT = 256;
		const uint rightshift = 32;
		const int total_len = dimensions * 32 / 8;
		const int long_steps = dimensions / 8;
		__local uint point_dim[1024];
		__local uchar result_byte_buffer[256];
		__local ulong result_buffer[256];
		__global long* result_as_long = (__global long*)&result[0];
		__global long* lookup_as_long = (__global long*)&lookup_table[0];


		if (get_global_id(0) == 0)
			prefetch(lookup_table, 256 * dimensions * dimensions);
	    for (uint id = get_group_id(0); id < N ; id+= get_num_groups(0))
		{
			uint input_offset =  id * dimensions;
			uint output_offset = id *total_len;

			for (int j = get_local_id(0); j < dimensions; j+= get_local_size(0))
				point_dim[j] = points[input_offset + j];


			for (int i_depth = 4; i_depth >= 1; --i_depth)
			{
				int result_offset = output_offset + 3 * dimensions - (i_depth - 1)*dimensions; // each mask has [dimensions] bytes in it
				int depth = (rightshift - 8 * i_depth);
				uint lr_offset = result_offset/8;

				for (int pos = get_local_id(0) ; pos < long_steps;  pos += get_local_size(0))
				{
					result_byte_buffer[pos] = 0;
					for (unsigned int d = 0; d < long_steps; ++d)
					{
						uint point_value = point_dim[d]; //points[input_offset + d]
						uint byte =  (point_value >> depth) & 0xFF;
						uint offset = d* dimensions * BYTE_COUNT + byte * dimensions;
						uint l_offset = offset/8;
						result_buffer[pos] |= lookup_as_long[l_offset+ pos ];
						barrier(CLK_LOCAL_MEM_FENCE);
					}
					result_as_long[lr_offset + pos] |=result_buffer[pos];
				}


				for (int pos = long_steps * 8 + get_local_id(0) ; pos < dimensions;  pos += get_local_size(0))
				{
					result_byte_buffer[pos] = 0;
					for (unsigned int d = 0; d < dimensions; ++d)
					{
						uint point_value = point_dim[d]; //points[input_offset + d]
						uint byte =  (point_value >> depth) & 0xFF;
						uint offset = d* dimensions * BYTE_COUNT + byte * dimensions;
						result_byte_buffer[pos] |= lookup_table[offset+ pos ];
						barrier(CLK_LOCAL_MEM_FENCE);
					}
					result[result_offset + pos] = result_byte_buffer[pos];
				}
				barrier(CLK_LOCAL_MEM_FENCE);

			}

		}
}
