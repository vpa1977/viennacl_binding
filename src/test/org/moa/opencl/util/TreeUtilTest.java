package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import org.junit.Test;
import org.moa.opencl.util.TreeUtil;

public class TreeUtilTest {

	@Test
	public void testParent() {
		assertEquals(TreeUtil.parent(7), 3);
		assertEquals(TreeUtil.parent(8), 3);
		
		assertEquals(TreeUtil.parent(9), 4);
		assertEquals(TreeUtil.parent(10), 4);
		assertEquals(TreeUtil.parent(3), 1);
		assertEquals(TreeUtil.parent(2), 0);
	}
	
	
	@Test
	public void testChild() {
		
		assertEquals(TreeUtil.child(3), 7);
		
		
		assertEquals(TreeUtil.child(4), 9);
		assertEquals(TreeUtil.child(1), 3);
		assertEquals(TreeUtil.child(0), 1);
	}


}
