#define SQRT_2 1.41421356237

__kernel void permute(const int size, __global const  DATA_TYPE * src, __global const int* perm, DATA_TYPE srhst_const, __global  DATA_TYPE * out)
{
	for (int i = get_local_id(0); i < size; i+= get_local_size(0))
		out[i] = SQRT_2 * srhst_const * src[ perm[i]];
};

__kernel void fix_nan(const int N, __global  DATA_TYPE * src)
{
	for (int i = get_local_id(0); i < N; i+= get_local_size(0))
	{
		ulong s = as_ulong(src[i]);
		bool is_nan= ((s & 0x7FF0000000000000) == 0x7FF0000000000000 ) || ((s & 0xFFF0000000000000) == 0xFFF0000000000000 );
		src[i] = select(src[i], 0.0, (ulong)is_nan);
	}
}
