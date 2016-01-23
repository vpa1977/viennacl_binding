package test.org.moa.gpu;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Test;
import org.moa.gpu.FJLT;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

public class FJLTTest {
	
	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}

	@Test
	public void testCreate() {
		Context mem_ctx = new Context(Context.Memory.MAIN_MEMORY , null );
		runFJLT(mem_ctx, 10 , 5);
		Context opencl_ctx = new Context(Context.DEFAULT_MEMORY , null );
		runFJLT(opencl_ctx, 10, 5);
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
	}
  
  public static void main(String[] args) throws Throwable
  {
     new FJLTTest(). testCreate();
  }

}
