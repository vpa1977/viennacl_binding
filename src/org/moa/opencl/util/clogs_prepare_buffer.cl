__kernel void prepare_buffer(__global uint* buffer)
{
	buffer[get_global_id(0)] = get_global_id(0);
}
