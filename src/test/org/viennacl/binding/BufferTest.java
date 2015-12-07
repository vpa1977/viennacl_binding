package test.org.viennacl.binding;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Test;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

public class BufferTest {

	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	@Test
	public void testCreateCPU() {
		Context ctx = new Context(Context.Memory.MAIN_MEMORY, null);
		Buffer buffer = new Buffer(ctx, 1024);
	}
	
	
	@Test
	public void testCreateGPU() {
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		Buffer buffer = new Buffer(ctx, 1024);
	}
	
	@Test
	public void testReadWriteBuffer() 
	{
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		testBufferReadWriteOnContext(ctx);
		ctx = new Context(Context.Memory.MAIN_MEMORY, null);
		testBufferReadWriteOnContext(ctx);
	}
	
	@Test
	public void testOpenclMapping()
	{
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		Buffer buffer = new Buffer(ctx, 1024);
		buffer.mapBuffer(Buffer.READ_WRITE, 2048, 2047);
		assertEquals(buffer.handle(), 0);
	}
	
	@Test 
	public void testCopyBuffer() 
	{
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);

		copyTest(ctx);
		ctx = new Context(Context.Memory.MAIN_MEMORY, null);
		copyTest(ctx);
		
	}


	private void copyTest(Context ctx) {
		Buffer buffer = new Buffer(ctx, 1024);
		buffer.mapBuffer(Buffer.READ_WRITE);
		buffer.write(0, 3234.90);
		buffer.write(DirectMemory.DOUBLE_SIZE, 3334.90);
		buffer.commitBuffer();
		assertEquals(buffer.handle(), 0);
		
		Buffer other = buffer.copy();
		other.mapBuffer(Buffer.READ_WRITE);
		double a = other.read(0);
		double b = other.read(DirectMemory.DOUBLE_SIZE);
		other.commitBuffer();
		assertEquals(a, 3234.90, 0.0001);
		assertEquals(b, 3334.90, 0.0001);
	}


	private void testBufferReadWriteOnContext(Context ctx) {
		Buffer buffer = new Buffer(ctx, 1024);
		
		buffer.mapBuffer(Buffer.READ_WRITE);
		
		DirectMemory.write(buffer.handle(), 3234.90);
		DirectMemory.write(buffer.handle()+ DirectMemory.DOUBLE_SIZE, 3334.90);
		buffer.commitBuffer();
		assertEquals(buffer.handle(), 0);
		
		double fixit = new Random().nextDouble();
		buffer.mapBuffer(Buffer.WRITE, DirectMemory.DOUBLE_SIZE * 4, 512);
		DirectMemory.write(buffer.handle(), fixit);
		buffer.commitBuffer();
		
		buffer.mapBuffer(Buffer.READ);
		double value1 = DirectMemory.read(buffer.handle());
		double value11 = DirectMemory.read(buffer.handle()+ DirectMemory.DOUBLE_SIZE);
		buffer.commitBuffer();
		assertEquals(3234.90, value1, 0.00001);
		assertEquals(3334.90, value11, 0.00001);
		
		buffer.mapBuffer(Buffer.READ, DirectMemory.DOUBLE_SIZE * 4, 512);
		double value2 = DirectMemory.read(buffer.handle());
		buffer.commitBuffer();
		assertEquals(value2, fixit, 0.00001);
	}


}
