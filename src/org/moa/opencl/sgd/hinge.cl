
__kernel void hinge(
		__global const VALUE_TYPE* class_values,
		__global const VALUE_TYPE* dot_product,
	  	__global const unsigned int * row_indices,
		__global const unsigned int * column_indices,
		__global const unsigned int * row_blocks,
		const unsigned int num_blocks,
		__global VALUE_TYPE * elements,
		__global VALUE_TYPE* loss
		  )
{
	unsigned int row_start = row_blocks[get_group_id(0)];
	unsigned int row_stop  = row_blocks[get_group_id(0) + 1];
	unsigned int rows_to_process = row_stop - row_start;
	unsigned int element_start = row_indices[row_start];
	unsigned int element_stop = row_indices[row_stop];

	for (unsigned int row = row_start; row < row_stop; ++row)
	{
		unsigned int current_row_start = row_indices[row];
		unsigned int current_row_stop  = row_indices[row + 1];
		unsigned int thread_base_id  = current_row_start + get_local_id(0);
		// sum whatever exceeds the current buffer:
		VALUE_TYPE labelScaled = 2 * class_values[row] - 1.0;
		VALUE_TYPE cond =labelScaled * dot_product[row];
		COND_TYPE condition = cond < 1;
		VALUE_TYPE factor = select(0.0, -labelScaled, condition );
		loss[row] = select(0.0, 1.0 - cond, condition);
		for (unsigned int j = thread_base_id; j < current_row_stop; j += get_local_size(0))
		{
			elements[j] = factor*elements[j];
		}
	}
}
