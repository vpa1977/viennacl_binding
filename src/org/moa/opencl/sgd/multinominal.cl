///

__kernel void multinominal_hinge(
				__global VALUE_TYPE* dot_products,
				const __global VALUE_TYPE* classes,
				const __global VALUE_TYPE* bias,
				const  int num_classes,
				const int rows)
{

	for (int row = get_global_id(0); row < rows; row += get_global_size(0))
	{
		int class_label = classes[row];
		int i = get_global_id(1);
		//for (int i = 0;i < num_classes; ++i)
		{
			if (get_global_size(1) == 1)
			{
				VALUE_TYPE y =class_label;
				VALUE_TYPE z = y  - (dot_products[i + row * num_classes] + bias[class_label]);
				y = 1;
				VALUE_TYPE upd = select( 0.0,y, (COND_TYPE)(z < 1));
				dot_products[i+ row * num_classes] = upd;
			}
			else
			{
				VALUE_TYPE y = select(-1, 1, class_label == i);
				VALUE_TYPE z = y * (dot_products[i + row * num_classes] + bias[class_label]);
				VALUE_TYPE upd = select( 0.0,y, (COND_TYPE)(z < 1));
				dot_products[i+ row * num_classes] = upd;
			}
		}
	}
}


__kernel void multiply_and_makedense(
		__global VALUE_TYPE* output,

	  	__global const unsigned int * row_indices,
		__global const unsigned int * column_indices,
		__global const unsigned int * row_blocks,
		const unsigned int num_blocks,
		__global VALUE_TYPE * elements,

		__global const VALUE_TYPE* dotProduct,
		const int index,
		const int num_columns)
{
	unsigned int row_start = row_blocks[get_group_id(0)];
	unsigned int row_stop  = row_blocks[get_group_id(0) + 1];
	unsigned int rows_to_process = row_stop - row_start;
	unsigned int element_start = row_indices[row_start];
	unsigned int element_stop = row_indices[row_stop];
	__local VALUE_TYPE multiplier;
	if (get_local_id(0) ==0)
		multiplier = dotProduct[index];
	barrier(CLK_GLOBAL_MEM_FENCE);
	for (unsigned int row = row_start; row < row_stop; ++row)
	{
		unsigned int current_row_start = row_indices[row];
		unsigned int current_row_stop  = row_indices[row + 1];
		unsigned int thread_base_id  = current_row_start + get_local_id(0);
		for (unsigned int j = thread_base_id; j < current_row_stop; j += get_local_size(0))
		{
			output[num_columns * row + column_indices[j] ] = elements[j]*multiplier;
		}
	}
}

__kernel void reduce_to_minibatch_dense(
		__global VALUE_TYPE* batch_update_matrix,
		__global const VALUE_TYPE* dot_product, // multiplier for weight update (before learning rate applied)
		 const int num_classes,
		 const int batch_size,
		 const int num_attributes,
		__global const VALUE_TYPE * elements)
{
	int lid = get_local_id(0);
	int thread_base_column = get_group_id(0)* get_local_size(0);
	if (thread_base_column >= num_attributes )
		return;

	int column = get_global_id(0);
	int dot_product_column = get_global_id(1);
	int class_offset = dot_product_column * num_attributes; // offset for the gradient
	__local VALUE_TYPE multiplier;
	__local VALUE_TYPE buffer[256];
	int row_start = 0;
	int row_end = batch_size;

	if (lid == 0)
		multiplier = dot_product[dot_product_column];
	buffer[lid] = 0;
	for (int row = row_start; row < row_end; ++row) // serially process rows
	{
		buffer[lid] += select(0.0, multiplier * elements[(int)mad((float)row,(float)num_attributes,(float)column)], (COND_TYPE)(column < num_attributes));
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	VALUE_TYPE upd = select(0.0, buffer[lid]/batch_size, (COND_TYPE)(column < num_attributes));
	int index = select(0, column, column< num_attributes);
	batch_update_matrix[class_offset +  index] += upd;
}

/*

__kernel void compute_margin_bounds(
		__global VALUE_TYPE* margins,
		const __global VALUE_TYPE* classes,
		const  int num_classes,
		const int rows,
		__global VALUE_TYPE* max_margin)
{

	for (int row = get_global_id(0); row < rows; row += get_global_size(0))
	{
		VALUE_TYPE max = -MAXFLOAT;
		for (int i = 0;i < num_classes; ++i)
		{
			double val =margins[i + row * num_classes];
			if (val > max )
				max = val;
		}
		int class_value = classes[row];
		VALUE_TYPE sum;
		for (int i = 0;i < num_classes; ++i)
		{
			margins[i] += select((VALUE_TYPE)0, -max, (COND_TYPE)(max > 0));
			sum += exp(select(margins[i], 0.0, (COND_TYPE)(i == class_value))
		}
		sum += select( exp(max), exp(-max), max > 0);
		if (max > 0)
		{
			// todo continue logistic regression implementation. will use hinge for now
		}
	}


}
*/
