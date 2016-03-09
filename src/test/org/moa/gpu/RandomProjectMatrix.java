package test.org.moa.gpu;

import static org.junit.Assert.*;

import org.junit.Test;
import org.moa.gpu.FloatFJLT;
import org.moa.gpu.MatrixRandomProjection;
import org.moa.opencl.util.BufHelper;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

public class RandomProjectMatrix {
	
	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	@Test
	public void testProject() 
	{
		int k = 2;
		int n = 20;
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		MatrixRandomProjection matrix = new MatrixRandomProjection(ctx, k,n);
		FloatFJLT fjlt = new FloatFJLT(ctx, k, n);
		
		float[] data = new float[ 2 * n];
		for (int i = 0;i < data.length; ++i)
			data[i] = 1;
		
		Buffer inputMatrix = new Buffer(ctx, 2 * n * DirectMemory.FLOAT_SIZE);
		Buffer outputMatrix = new Buffer(ctx, 2 * k * DirectMemory.FLOAT_SIZE);
		
		inputMatrix.mapBuffer(Buffer.WRITE);
		inputMatrix.writeArray(0,  data);
		inputMatrix.commitBuffer();
		
		matrix.project(inputMatrix, 2, outputMatrix);
		
		
		float[] result = BufHelper.rbf(outputMatrix);
		
		fjlt.transform(inputMatrix, 2, outputMatrix);
		
		float[] f_result = BufHelper.rbf(outputMatrix);
		
		matrix.project(inputMatrix, outputMatrix);
		
		float[] res2 = BufHelper.rbf(outputMatrix);
		
		float[] first = matrix.softProject(data);
		for (int i = 0; i < k; ++i)
		{
			assertEquals(first[i], result[i], 0.001);
		}
	}
}
