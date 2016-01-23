package test.org.moa.opencl.kdtree;

import static org.junit.Assert.*;

import org.junit.Test;

public class NodeNumberTest {

	
	private long nextPow2(long v)
	{
		  v |= v >> 1;
		  v |= v >> 2;
	    v |= v >> 4;
	    v |= v >> 8;
	    v |= v >> 16;
	    v++;
	    return v;
		
	}
	
	public long level(long node_num) 
	{
		if (node_num == 0) return 0;
		long next = nextPow2(node_num);
		long kplusone = (long)(Math.log(next)/Math.log(2));
		return kplusone - 1;
	}
	
	public long max_level_id(long level)
	{
		if (level == 0) return 0;
		return (int)Math.pow(2, level+1) - 2;
	}
	
	public long child(long id)
	{
		long level = level(id);
		long max_prev_level = max_level_id(level-1);
		long max_cur_level = max_level_id(level);
		long count = id - max_prev_level;
		return count *2 + max_cur_level -1;
	}
	
	
	@Test
	public void test() 
	{
		assertEquals(max_level_id(1), 2);
		for (int l = 1; l < 10 ; ++l)
		{
			int lmin = (int)Math.pow(2, l) - 1;
			int lmax = (int)Math.pow(2, l+1) - 1;
			for (int i = lmin+1; i <lmax ; ++i)
				assertEquals("for level + " + l + " node num " + i, level(i), l);
		}
		
		assertEquals(child(7), 15);
		assertEquals(child(3), 7);
	}

}

