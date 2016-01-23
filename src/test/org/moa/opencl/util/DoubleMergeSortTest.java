package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import org.junit.Test;
import org.moa.opencl.util.DoubleMergeSort;
import org.moa.opencl.util.Operations;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

public class DoubleMergeSortTest {

	
	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
	@Test
	public void testCreate() {
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		
		DoubleMergeSort ms = new DoubleMergeSort(ctx, 1000);
		Operations ops = new Operations(ctx);
    
		Buffer stuffToSortIndices = new Buffer(ctx, DirectMemory.INT_SIZE * 1000);
    ops.prepareOrderKey(stuffToSortIndices, 1000);
		Buffer stuffToSort = new Buffer(ctx, DirectMemory.DOUBLE_SIZE * 1000);
		double[] input = new double[1000];
		for (int i = 0;i < input.length ; ++i)
			input[i] = input.length - i;
		stuffToSort.mapBuffer(Buffer.WRITE);
		stuffToSort.writeArray(0, input);
		stuffToSort.commitBuffer();
		
		ms.sort(stuffToSort, stuffToSortIndices);
		
		int[] indices = new int[1000];
		stuffToSortIndices.mapBuffer(Buffer.READ);
		stuffToSortIndices.readArray(0, indices);
		stuffToSortIndices.commitBuffer();
		for (int i = 0;i < input.length ; ++i)
			assertEquals(indices[i], input.length -i -1);
	}

  public static void main(String[] args)
  {
    int sort_size = 600;
    Context ctx = new Context(Context.Memory.HSA_MEMORY, null);
		Operations ops = new Operations(ctx);
		DoubleMergeSort ms = new DoubleMergeSort(ctx, sort_size);
    double[] input = new double[sort_size];
		for (int i = 0;i < input.length ; ++i)
			input[i] = input.length - i;

    Buffer stuffToSortIndices = new Buffer(ctx, DirectMemory.INT_SIZE * sort_size);
    ops.prepareOrderKey(stuffToSortIndices, 10);
		Buffer stuffToSort = new Buffer(ctx, DirectMemory.DOUBLE_SIZE * sort_size);
		stuffToSort.mapBuffer(Buffer.WRITE);
		stuffToSort.writeArray(0, input);
		stuffToSort.commitBuffer();
		ms.sort(stuffToSort, stuffToSortIndices);
    
 		int[] indices = new int[sort_size];
		stuffToSortIndices.mapBuffer(Buffer.READ);
		stuffToSortIndices.readArray(0, indices);
		stuffToSortIndices.commitBuffer();
		for (int i = 0;i < input.length ; ++i)
			assertEquals(indices[i], input.length -i -1);

  }
}
