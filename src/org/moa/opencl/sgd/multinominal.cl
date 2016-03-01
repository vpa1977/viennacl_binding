///

__kernel void multinominal_hinge(
				__global VALUE_TYPE* dot_products,
				const __global VALUE_TYPE* classes,
				const __global VALUE_TYPE* weights,
				const int classIndex,
				const  int num_classes,
				const int rows,
				const int num_attributes)
{

	for (int row = get_global_id(0); row < rows; row += get_global_size(0))
	{
		int i = get_global_id(1);
		VALUE_TYPE bias = weights[ i *num_attributes + classIndex];

		int class_label = classes[row];
		//for (int i = 0;i < num_classes; ++i)
		{

			VALUE_TYPE y = select(-1, 1, class_label == i);
			VALUE_TYPE z = y * (dot_products[i + row * num_classes] + bias);
			VALUE_TYPE upd = select( 0.0,y, (COND_TYPE)(z < 1));
			dot_products[i+ row * num_classes] = upd;

		}
	}
}



__kernel void reduce_to_minibatch_dense(
		__global VALUE_TYPE* batch_update_matrix,
		__global const VALUE_TYPE* dot_product, // multiplier for weight update (before learning rate applied)
		 const int num_classes,
		 const int row_end,
		 const int num_attributes,
		__global const VALUE_TYPE * elements,
		const int class_attribute_index)
{
	int dot_product_column = get_global_id(1);
	int lid = get_local_id(0);
	int column = get_global_id(0);
	__local VALUE_TYPE buffer[256];
	__local VALUE_TYPE multipliers[256];
	for (int id  = get_local_id(0); id < row_end; id += get_local_size(0))
		multipliers[id] = dot_product[dot_product_column + id * num_classes]/row_end;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (;column < num_attributes; column += get_global_size(0))
	{
		int class_offset = dot_product_column * num_attributes; // offset for the gradient
		buffer[lid] = 0;
		for (int row = 0; row < row_end; ++row) // serially process rows
		{
			buffer[lid] += multipliers[row] * select(1.0, elements[row * num_attributes + column], (COND_TYPE)(column != class_attribute_index));
		}
		batch_update_matrix[class_offset +  column] = buffer[lid];
	}
}



__kernel void reduce_to_minibatch_sparse(
		__global VALUE_TYPE* batch_update_matrix,
		__global const VALUE_TYPE* dot_product, // multiplier for weight update (before learning rate applied)
		 const int num_classes,
		 const int num_attributes,
		__global const unsigned int * row_indices,
		__global const unsigned int * column_indices,
		__global const unsigned int * row_blocks,
		const unsigned int num_blocks,
		__global VALUE_TYPE * elements,
		const int row,
		const int num_rows,
		const int class_attribute_index)
{
	int class_index = get_global_id(1);
	int class_offset = class_index * num_attributes; // offset for the gradient
	int element_index  = row_indices[row] + get_global_id(0);
	int row_end = row_indices[row + 1];
	__local VALUE_TYPE multiplier;
	if (get_local_id(0) ==0)
		multiplier= dot_product[class_index + num_classes * row]/num_rows;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (; element_index < row_end; element_index += get_global_size(0))
	{
		int column = column_indices[ element_index];
		batch_update_matrix[class_offset + column ] += select(1.0, elements[element_index], (COND_TYPE)(column !=class_attribute_index)) * multiplier;
	}

}
