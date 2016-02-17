package org.moa.opencl.util;
/** 
 * Balanced binary tree utils
 * @author john
 *
 */
public class TreeUtil {
	public static long nextPow2(long v) {
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v++;
		return v;
	}

	public static long level(long node_num) {
		if (node_num == 0)
			return 0;
		long next = nextPow2(node_num);
		long kplusone = (long) (Math.log(next) / Math.log(2));
		return kplusone - 1;
	}

	public static long max_level_id(long level) {
		if (level == 0)
			return 0;
		return (int) Math.pow(2, level + 1) - 2;
	}

	/** 
	 * left child of id. 
	 * right would be child() + 1
	 * @param id
	 * @return
	 */
	public static long child(long id) {
		long level = level(id);
		long max_prev_level = max_level_id(level - 1);
		long max_cur_level = max_level_id(level);
		long count = id - max_prev_level;
		return count * 2 + max_cur_level - 1;
	}
	
	/** 
	 * parent for given id
	 * @param id
	 * @return
	 */
	public static long parent(long id)
	{
		long level = level(id) - 1;
		long max_prev_level = max_level_id(level - 1);
		long max_cur_level = max_level_id(level);
		
		long count = id - max_prev_level;
		
		if ( (id & 0x1) != 0)
		{
			++id;
		}
	    return (id - max_cur_level + 2*max_prev_level) /2; 
	}
	
	public static boolean isLeft(long id)
	{
		return ( (id & 0x1) != 0);
	}

	public static int rightChild(int node_idx) {
		return 2 * node_idx + 2;
	}
	
	public static int leftChild(int node_idx) {
		return 2 * node_idx + 1;
	}

	
}
