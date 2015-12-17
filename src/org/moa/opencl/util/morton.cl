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
