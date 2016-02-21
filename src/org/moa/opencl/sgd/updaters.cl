__kernel void simple_update(
		__global  const VALUE_TYPE* gradients,
		 const int num_classes,
		 const int num_columns,
		__global const VALUE_TYPE* tau,
		__global  VALUE_TYPE* Es,
		__global  VALUE_TYPE* El,
		__global int* weights_delta,
		const int batch_number)
{
	int column = get_global_id(0);
	int row = get_global_id(1);
	int error_offset =batch_number*num_classes *num_columns+ num_columns*row;
	for (; column < num_columns; column += get_global_size(0))
	{
		VALUE_TYPE gradient = gradients[row *num_columns+ column] +Es[error_offset+ column] +  El[error_offset+ column];
		VALUE_TYPE abs_grad = copysign(gradient, 1.0);
		if (abs_grad +0.000000000001>= tau[column + row*num_classes])
		{
			atomic_add((volatile __global  int*)&weights_delta[row * num_columns + column], (int)copysign(1,gradient));
			abs_grad -= tau[column];
			El[error_offset+ column] = sign(gradient)*abs_grad;
			Es[error_offset+ column] = 0;
		}
		else
		{
			El[error_offset+ column] = 0;
			Es[error_offset+ column] = gradient;
		}
	}
}

__kernel void read_weights(__global VALUE_TYPE* weights, __global  const VALUE_TYPE* old_weights, int columns, int rows,__global int* weights_delta,
		__global const VALUE_TYPE* tau, const VALUE_TYPE learning_rate)
{
	for (int j = get_global_id(0); j < columns * rows;  j+= get_global_size(0))
	{
		weights[j] = old_weights[j] + learning_rate *tau[j] * weights_delta[j];
	}
}


__kernel void apply_delta(__global  VALUE_TYPE* old_weights, int columns, int rows,__global int* weights_delta,__global const VALUE_TYPE* tau, const VALUE_TYPE learning_rate)
{
	for (int j = get_global_id(0); j < columns * rows;  j+= get_global_size(0))
	{
		old_weights[j] = old_weights[j] + learning_rate* tau[j] * weights_delta[j];
		weights_delta[j] = 0;
	}
}

__kernel void update_tau(	__global  VALUE_TYPE* tau,
		__global const VALUE_TYPE* Es,
		__global const VALUE_TYPE* El,
		int num_batches, int columns, int rows)
{

	for (int j = get_global_id(0); j < columns * rows;  j+= get_global_size(0))
	{
		VALUE_TYPE sum = 0;
		for (int batch = 0; batch < num_batches; ++ batch)
		{
			sum += select(0.0, -(tau[j] - fabs(Es[batch * columns*rows + j])),
					(COND_TYPE)(Es[batch * columns*rows + j]!=0))
				   +		 fabs(El[batch*columns*rows +j]);
		}
		sum = sum / num_batches;
		tau[j] += sum;
	}
}
