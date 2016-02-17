package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import java.util.Arrays;

import org.junit.Test;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.moa.opencl.util.DoubleBitonicSort;
import org.moa.opencl.util.DoubleMergeSort;
import org.omg.CosNaming.BindingIteratorOperations;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

public class BenchmarkRadixSort {
	
	static
	{
		System.loadLibrary("viennacl-java-binding");
	}
  
  public static void main(String[] args)
  {
    new BenchmarkRadixSort().testRadixSort();
  }
	
	@Test
	public void testRadixSort()
	{
		Context ctx = new Context(Context.DEFAULT_MEMORY, null);
		double[] keys = new double[1024];
		for (int i = 0;i < keys.length; ++i)
		{
			keys[i] = Math.exp(-i);
		}
		
		Buffer the_keys = new Buffer(ctx, keys.length* DirectMemory.DOUBLE_SIZE);
		the_keys.mapBuffer(Buffer.WRITE);
		the_keys.writeArray(0, keys);
		the_keys.commitBuffer();
		
		CLogsVarKeyJava sort_benchmark = new CLogsVarKeyJava(ctx, false, "unsigned long", null);
		
		sort_benchmark.sortFixedBuffer(the_keys, null, keys.length);
		
		the_keys.mapBuffer(Buffer.READ);
		the_keys.readArray(0, keys);
		the_keys.commitBuffer();
		
		System.out.println("");
		
	}

	
	public void test() {
		Context ctx = new Context(Context.DEFAULT_MEMORY, null);
		CLogsVarKeyJava sort_benchmark = new CLogsVarKeyJava(ctx, false, "unsigned long", null);
		DoubleMergeSort sort_merge  = new DoubleMergeSort(ctx, 134217728);
		DoubleBitonicSort sort_bitonic = new DoubleBitonicSort(ctx, 134217728);
		int size = 1024;
		for (size = 512; size <= 134217728; size*= 2)
		{
			Buffer indices = new Buffer(ctx, size* DirectMemory.INT_SIZE);
			Buffer the_keys = new Buffer(ctx, size* DirectMemory.LONG_SIZE);
			long[] keys = new long[size];
			for (int i = 0; i < keys.length; ++i)
				keys[i] = (long)(Math.random() * 100000);
			
			
			double duration_sort =0;
			for (int i = 0; i < 1 ; ++i)
			{
				the_keys.mapBuffer(Buffer.WRITE);
				the_keys.writeArray(0, keys);
				the_keys.commitBuffer();
				//long[] lim = new long[keys.length];
				//System.arraycopy(keys, 0, lim, 0, keys.length);
				
				long start = System.nanoTime();
				sort_benchmark.sortFixedBuffer(the_keys, null, size);
				//sort_merge.sort(the_keys, indices);
				//
				//Arrays.sort(lim);
				long end = System.nanoTime();
				duration_sort += (end - start);
			}
			long start = System.nanoTime();
			ctx.finishDefaultQueue();
			long end = System.nanoTime();
			duration_sort += (end - start);
			
			duration_sort = duration_sort / 1000000L; //milliseconds
			duration_sort = duration_sort / 1000; // seconds.
			System.out.println(size + " \t " + (duration_sort/1));
		}
	}

}
