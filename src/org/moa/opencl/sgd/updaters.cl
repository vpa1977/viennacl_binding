
// local_size(256,1)
// global_size
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

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
		__global VALUE_TYPE * EsAvg,
		__global VALUE_TYPE * ElAvg,
		__global const VALUE_TYPE* Es,
		__global const VALUE_TYPE* El,
		int num_batches, int columns, int rows,
		const int update)
{

	for (int j = get_global_id(0); j < columns * rows;  j+= get_global_size(0))
	{
		VALUE_TYPE sumL = 0;
		VALUE_TYPE sumR = 0;
		for (int batch = 0; batch < num_batches; ++ batch)
		{
			sumL += fabs(Es[batch * columns*rows + j]);
			sumR += tau[j] + fabs(El[batch*columns*rows +j]);
		}
		sumL = sumL / num_batches;
		sumR = sumR / num_batches;
		EsAvg[j] = ((update - 1) * EsAvg[j] + sumL) / update;
		ElAvg[j] = ((update - 1) * ElAvg[j] + sumR) / update;

		tau[j] = (ElAvg[j] - EsAvg[j])/2;
	}
}


__kernel void simple_update2(
		__global  const VALUE_TYPE* gradients,
		 const int num_classes,
		 const int num_columns,
		__global const VALUE_TYPE* tau,
		__global  VALUE_TYPE* Es,
		__global  VALUE_TYPE* El,
		const int batch_number,
		const VALUE_TYPE learning_rate,
		__global VALUE_TYPE* weights,
		__global VALUE_TYPE* weights_delta)
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
			VALUE_TYPE old = weights[row * num_columns + column];
			VALUE_TYPE sum = old +learning_rate *  gradient;
			volatile ulong test = atom_cmpxchg((volatile __global ulong*)(&weights[row * num_columns + column]),as_ulong(old), as_ulong(sum));
			if (test!= old)
			{
				atomic_add((volatile __global  int*)&weights_delta[row * num_columns + column], (int)copysign(1,gradient));
				abs_grad -= tau[column];
				El[error_offset+ column] +=select(0.0,  sign(gradient)*abs_grad,(ulong) (test!=old));
			}
			Es[error_offset+ column] = 0;
		}
		else
		{
			El[error_offset+ column] = 0;
			Es[error_offset+ column] = gradient;
		}
	}
}

