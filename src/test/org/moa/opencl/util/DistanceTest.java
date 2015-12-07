package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import org.junit.Test;
import org.moa.opencl.util.Distance;
import org.moa.opencl.util.MinMax;
import org.viennacl.binding.Context;

public class DistanceTest {

	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
	@Test
	public void testCreate() {
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		Distance d = new Distance(ctx);
		assertTrue(d != null);
	}

}
