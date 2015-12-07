package test.org.viennacl.binding;

import static org.junit.Assert.*;

import org.junit.Test;
import org.viennacl.binding.Context;
import org.viennacl.binding.Kernel;
import org.viennacl.binding.Queue;

public class ContextTest {

	static {
		System.loadLibrary("viennacl-java-binding");
	}
	@Test
	public void testContext() {
		Context ctx = new Context(Context.Memory.MAIN_MEMORY, null);
		assertTrue(ctx != null);
		ctx.release();
		ctx = new Context(Context.Memory.OPENCL_MEMORY, "-cl-std=CL2.0 -D CL_VERSION_2_0");
		Queue newQueue = ctx.createQueue();
		assertTrue(ctx != null);
		assertTrue(newQueue != null);
		ctx.release();
	}

	@Test
	public void testAddProgramAndGetKernel() {
		String program = "__kernel void assignX(__global long* src, __global  long* dst) { src[0] = dst[0]; }";
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, "-cl-std=CL2.0 -D CL_VERSION_2_0");
		ctx.add("simple_program", program);
		Kernel newKernel = ctx.getKernel("simple_program", "assignX");
		assertTrue(newKernel != null);
		ctx.removeProgram("simple_program");
		ctx.release();
	}


}
