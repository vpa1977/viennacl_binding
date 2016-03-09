
#define WARP_SIZE 64
#define KERNEL(size) __kernel __attribute__((reqd_work_group_size(size, 1, 1)))

enum DivergenceKind
{
	dkNONE,
	dkLESS,
	dkGREATER,

};
//KERNEL(WARP_SIZE)
__kernel
void search(__global const uchar* keys, __global const uint* key_indices, __global const uchar* searchKey, int keyLength, __global uint* searchBounds, const uint natural)
{
	int lowBound = searchBounds[0]; // global lowBound
	int highBound = searchBounds[1]; // global highBound

	if (highBound - lowBound < 2) // lowBound follows highBound or the same
		return;



	int group_count = get_num_groups(0);
	int group_size = 1+ (highBound - lowBound) / group_count;
	int localLowBound = lowBound + get_group_id(0)* group_size;
	if (localLowBound >= highBound)
		return;
	int localHighBound = min(highBound, localLowBound + group_size);
	if (localHighBound + group_size >= highBound)
		localHighBound = highBound;

	volatile __local int divergencePositionLow;
	volatile __local int divergencePositionHigh;
	__local int divergenceKindLow;
	__local int divergenceKindHigh;

	if (get_local_id(0) == 0)
	{
//		atomic_xchg(&divergencePositionLow, keyLength);
//		atomic_xchg(&divergencePositionHigh, keyLength);
		divergencePositionLow =  keyLength;
		divergencePositionHigh = keyLength;
		divergenceKindLow = dkNONE;
		divergenceKindHigh = dkNONE;
	}
	barrier(CLK_LOCAL_MEM_FENCE); // to avoid further divergence
	int lowIndex = key_indices[localLowBound] * keyLength;
	int highIndex = key_indices[localHighBound] * keyLength;
	int pos = get_local_id(0);
	bool next_cycle = pos < keyLength && (divergenceKindLow == dkNONE || divergenceKindHigh == dkNONE);
	while (next_cycle )
	{
		int key_offset = natural ? pos : (keyLength -1-pos);
		uchar term  =  searchKey[key_offset]; // preload keys
		uchar keyLow  =keys[key_offset+ lowIndex];
		uchar keyHigh  =keys[key_offset+ highIndex];

		if (keyLow < term)
		{
			atomic_min(&divergencePositionLow,pos);
			if (divergencePositionLow == pos)
				divergenceKindLow = dkLESS;
		}

		if (keyLow > term)
		{
			atomic_min(&divergencePositionLow,pos);
			if (divergencePositionLow == pos)
				divergenceKindLow = dkGREATER;
		}

		if (keyHigh< term)
		{
			atomic_min(&divergencePositionHigh,pos);
			if (divergencePositionHigh == pos)
				divergenceKindHigh = dkLESS;
		}

		if (keyHigh > term)
		{
			atomic_min(&divergencePositionHigh,pos);
			if (divergencePositionHigh == pos)
				divergenceKindHigh = dkGREATER;
		}

		barrier(CLK_LOCAL_MEM_FENCE); // to avoid further divergence
		pos+= WARP_SIZE;
		next_cycle = pos < keyLength && (divergenceKindLow == dkNONE || divergenceKindHigh == dkNONE);
	}
	if (get_local_id(0) == 0)
	{
		// repeat for this interval
		if (divergenceKindHigh == dkGREATER && divergenceKindLow == dkLESS)
		{
			searchBounds[0] = localLowBound;
			searchBounds[1] = localHighBound;
		}

		// value find
		if (divergenceKindHigh == dkNONE)
		{
			searchBounds[0] = localHighBound;
			searchBounds[1] = localHighBound;
		}

		if (divergenceKindLow == dkNONE)
		{
			searchBounds[0] = localLowBound;
			searchBounds[1] = localLowBound;
		}



		if (localHighBound == highBound  && (divergenceKindHigh == dkLESS))
		{
			searchBounds[0] = localHighBound;
			searchBounds[1] = localHighBound;
			searchBounds[2] = dkGREATER;
		}

		if ((get_group_id(0) == 0) && (divergenceKindLow == dkGREATER))
		{
			searchBounds[0] = localLowBound;
			searchBounds[1] = localLowBound;
			searchBounds[2] = dkLESS;
		}

	}



}
