package test.org.moa.gpu;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Test;
import org.moa.gpu.FJLT;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.filters.unsupervised.attribute.RandomProjection;

public class FJLTTest {
	
	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}

	@Test
	public void testCreate() {
		Context mem_ctx = new Context(Context.Memory.MAIN_MEMORY , null );
		runFJLT(mem_ctx, 738 ,(int) Math.sqrt(738));
		Context opencl_ctx = new Context(Context.DEFAULT_MEMORY , null );
		runFJLT(opencl_ctx, 738 , (int)Math.sqrt(738));
	}
	
	@Test 
	public void testCreateBatch()
	{
		Context ctx = new Context(Context.DEFAULT_MEMORY , null );
		FJLT fjlt = new FJLT(ctx,  5, 3);
		Buffer src = new Buffer(ctx, 10 * DirectMemory.DOUBLE_SIZE);
		double[] src1 = new double[]{ 1,1,1,1,1,0,0,0,0,0};
		double[] src2 = new double[]{ 1,1,1,1,1,4,4,4,4,4};
		double[] test1 = new double[6];
		double[] test2 = new double[6];
		
		Buffer dst = new Buffer(ctx, 6 * DirectMemory.DOUBLE_SIZE);
		
		src.mapBuffer(Buffer.WRITE);
		src.writeArray(0, src1);
		src.commitBuffer();
		
		fjlt.transform(src, 2, dst);
		
		dst.mapBuffer(Buffer.READ);
		dst.readArray(0, test1);
		dst.commitBuffer();
		
		src.mapBuffer(Buffer.WRITE);
		src.writeArray(0, src2);
		src.commitBuffer();
		
		fjlt.transform(src, 2, dst);
		
		dst.mapBuffer(Buffer.READ);
		dst.readArray(0, test2);
		dst.commitBuffer();
		
		for (int i = 0; i < 3; ++i)
			assertEquals(test1[i], test2[i], 0.0001);
		
	}

	private void runFJLT(Context ctx, int i, int j) {
		FJLT fjlt = new FJLT(ctx,  i, j);
		Buffer src = new Buffer(ctx, i * DirectMemory.DOUBLE_SIZE);
		Buffer dst = new Buffer(ctx, j * DirectMemory.DOUBLE_SIZE);
		double[] random_src = new double[i];
		Random rnd = new Random();
		for (int k = 0; k < random_src.length; ++k)
		{
			random_src[k] = rnd.nextDouble(); 
		}
		src.mapBuffer(Buffer.WRITE);
		src.writeArray(0, random_src);
		src.commitBuffer();
		
		fjlt.transform(src, dst);
		
		
		long start = System.nanoTime();
		for (int t = 0; t < 1000 ; ++t)
			fjlt.transform(src, dst);
		double[] temp = new double[5];
		dst.mapBuffer(Buffer.READ);
		dst.readArray(0, temp);
		dst.commitBuffer();
		long end = System.nanoTime();

		System.out.println("Transform time " + (end-start)/1000000.0 + " msec per 1000");
	}
  
  public static void main(String[] args) throws Throwable
  {
     new FJLTTest(). testCreate();
  }

}
