#define RAW_OFFSET(x) ( code_length - x)

// global_size(1) == 3
__kernel void morton_code_backup(__global uchar* result,
													__global uchar* lookup_table,
													__global uint* points,
													const uint dimensions,
													const uint N)
{
		const uint BYTE_COUNT = 256;
		const uint rightshift = 32;
		const uint code_length = dimensions * 4 -1;
		const int total_len = dimensions * 32 / 8;

	  for (uint id = get_global_id(0); id < N ; id+= get_global_size(0))
		{
			uint input_offset =  id * dimensions;
			uint output_offset = id *total_len;

			for (uint i_depth = 1; i_depth <= 4; ++i_depth)
			{
					int result_offset = 3 * dimensions - (i_depth - 1)*dimensions; // each mask has [dimensions] bytes in it
					for (int pos = result_offset + dimensions-1;  pos >= result_offset ; --pos)
					{
						for (unsigned int d = 0; d < dimensions; ++d)
						{
							uint point_value = points[input_offset + d];
							uint shifted_value =  point_value >> (rightshift - 8 * i_depth);
							uint byte = shifted_value & 0xFF;
							uint offset = d* dimensions * BYTE_COUNT + byte * dimensions + pos - result_offset;
							result[output_offset + RAW_OFFSET(pos)] |= lookup_table[offset];
						}
					}
				}

		}
}


__kernel void morton_code_group_backup(__global uchar* result,
													__global const uchar* lookup_table,
													__global uint* points,
													const uint dims,
													const uint N)
{
		const uint BYTE_COUNT = 256;
		const uint rightshift = 32;
		int id = get_global_id(0);
		int input_offset =  id /(dims*  4);
		input_offset *= dims;
		int pos = id % (dims *4);
		int i_depth = 1  + pos/dims;
		int code_pos = pos % dims; // current offset in dims code.
		for (int d = 0; d < dims; ++d)
		{
			int b = (points[input_offset+d] >> (rightshift - 8 * i_depth)) & 0xFF;
			int offset = d* dims * BYTE_COUNT + b * dims + dims-1 - code_pos;
			result[id] |= lookup_table[offset];
		}

}

// 1 thread - 1 int
__kernel void morton_code_group(__global uint* result,
													__global const uchar* lookup_table,
													__global uint* points,
													const uint dims,
													const uint N)
{

		const uint BYTE_COUNT = 256;
		const uint rightshift = 32;
		int update_count = dims / 8;
		int id = get_global_id(0)*4;
		int input_offset =  get_global_id(0) /dims;
		input_offset *= dims;

		//int top_dim = select( dims , dims - 8*(id % update_count) , update_count !=0 );
		//int low_dim = select(0 , top_dim -8, update_count !=0 );
		//top_dim = select( dims , dims - 8*((id+4) % update_count) , update_count !=0 );

		//int start = min(top_dim, low_dim);
		//int end = max(top_dim, low_dim);

		uint sum = 0;
		for (int d = 0; d < dims; ++d)
		{
			uint point = points[input_offset + d];
			int slot = d* dims * BYTE_COUNT;
			for (int cache_pos =0; cache_pos < 4; ++cache_pos)
			{
				int pos = (id + cache_pos) % (dims *4);
				int i_depth = 1  + pos/dims;
				int code_pos = pos % dims; // current offset in dims code.
				int b = (point >> (rightshift - 8 * i_depth)) & 0xFF;
				int offset = slot + (b +1)* dims -1 - code_pos;
				sum |=  lookup_table[offset] << (cache_pos *8);
			}
		}
		result[get_global_id(0)] = sum;
}



__kernel void morton_code_group_new(__global uint* result,
													__global const uchar* lookup_table,
													__global uint* points,
													const uint dims,
													const uint N)
{
	volatile __local uint local_result[1024];
	__local uint point_cache[1024];
	int input_offset =  get_group_id(0)*dims;
	for (int i = get_local_id(0); i < dims; i+= get_local_size(0))
	{
		point_cache[i] = points[input_offset + i];
		local_result[i] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	const uint BYTE_COUNT = 256;
	const uint rightshift = 32;
	int update_count = dims / 8;
	for (int gid = get_local_id(0); gid < dims; gid += get_local_size(0)) // each output int - separate thread
	{
		int id = gid * 4; // overall byte position
		int top_dim = select( dims , dims - 8*(id % update_count) , update_count !=0 );
		int low_dim = select(0 , top_dim -8, update_count !=0 );
		top_dim = select( dims , dims - 8*((id+4) % update_count) , update_count !=0 );

		int start = min(top_dim, low_dim);
		int end = max(top_dim, low_dim);

		uint sum = 0;
		for (int d = low_dim ; d < top_dim; ++d)
		{
			uint point = point_cache[d];
			int slot = d* dims * BYTE_COUNT;
			for (int cache_pos =0; cache_pos < 4; ++cache_pos)
			{
				int pos = (id + cache_pos) % (dims *4);
				int i_depth = 1  + pos/dims;
				int code_pos = pos % dims; // current offset in dims code.
				int b = (point >> (rightshift - 8 * i_depth)) & 0xFF;
				int offset = slot + (b +1)* dims -1 - code_pos;
				atomic_or(&local_result[gid], (lookup_table[offset] << (cache_pos *8)));
			}
		}
		result[input_offset + gid] = local_result[id];
	}


}
