package test.org.viennacl.binding;

import static org.junit.Assert.*;

import org.junit.Test;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

public class KernelTest {
	
	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}

	@Test
	public void testKernelAssign() {
		String program = "__kernel void assignX(__global long* src, __global  long* dst) { dst[0] = src[0]; }";
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, "-cl-std=CL2.0 -D CL_VERSION_2_0");
		ctx.add("simple_program", program);
		Kernel newKernel = ctx.getKernel("simple_program", "assignX");

		Buffer left = new Buffer(ctx, 1024);
		Buffer right = new Buffer(ctx, 1024);
		
		left.mapBuffer(Buffer.WRITE);
		DirectMemory.writeLong(left.handle(), 1337);
		left.commitBuffer();
		
		newKernel.set_global_size(0, 256);
		newKernel.set_arg(0, left);
		newKernel.set_arg(1, right);
		newKernel.invoke();
		
		right.mapBuffer(Buffer.READ);
		long value = DirectMemory.readLong(right.handle());
		right.commitBuffer();
		assertEquals(value, 1337);
		
		assertTrue(newKernel != null);
		ctx.freePrograms();

	}

}
