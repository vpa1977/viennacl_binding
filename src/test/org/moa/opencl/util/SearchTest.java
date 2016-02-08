package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import java.math.BigInteger;
import java.util.Arrays;

import org.junit.Test;

import org.moa.opencl.util.NarySearch;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

public class SearchTest {

	
	static {
		System.loadLibrary("viennacl-java-binding");
	}
	
	@Test
	public void testAtomicMin() 
	{
		String sampleTest = "__kernel void testAtomicMin(__global uint* testval) "
		+ "{"
		+ "volatile __local int divergencePositionLow;"
		+ "atomic_xchg(&divergencePositionLow, 10);"
		+ " testval[0] = divergencePositionLow;"
		+ " atomic_min(&divergencePositionLow, 5);"
		+ " testval[1] = divergencePositionLow;"
		+ "}";
		Context ctx = new Context( Context.DEFAULT_MEMORY, null);
		Buffer keyValues = new Buffer(ctx, 2*DirectMemory.INT_SIZE);
		ctx.add("sample", sampleTest);
		Kernel k = ctx.getKernel("sample", "testAtomicMin");
		
		k.set_arg(0, keyValues);
		k.set_local_size(0, 64);
		k.set_global_size(0, 64);
		k.invoke();
		
		int[] test = new int[2];
		keyValues.mapBuffer(Buffer.READ);
		keyValues.readArray(0, test);
		keyValues.commitBuffer();
		
		assertEquals(test[0], 10 );
		assertEquals(test[1], 5 );
	}
	
	
	@Test
	public void testCreate() {
		int key_len = (int)DirectMemory.INT_SIZE;
		int seq_len = 1000;
		
		int[] order = new int[seq_len];
		int[] values = new int[seq_len];
		for (int i = 0;i < seq_len; ++i)
		{
			order[i] = i;
			values[i] = i+1;
		}
		
		
		
		Context ctx = new Context( Context.DEFAULT_MEMORY, null);
		Buffer keyValues = new Buffer(ctx, seq_len* key_len);
		Buffer keyOrder = new Buffer(ctx, seq_len * DirectMemory.INT_SIZE);
		
		keyValues.mapBuffer(Buffer.WRITE);
		keyValues.writeArray(0,  values);
		keyValues.commitBuffer();
		
		keyOrder.mapBuffer(Buffer.WRITE);
		keyOrder.writeArray(0,  order);
		keyOrder.commitBuffer();
		
		Buffer searchTerm = new Buffer(ctx, key_len);
		
		
		NarySearch search = new NarySearch(ctx, false);
		
		searchTerm.mapBuffer(Buffer.WRITE);
		searchTerm.writeInt(0, 0xFFFF);
		searchTerm.commitBuffer();

		search.search(keyValues, keyOrder, searchTerm, key_len, seq_len, false);
		assertEquals(search.getSearchPos(), 999);

		searchTerm.mapBuffer(Buffer.WRITE);
		searchTerm.writeInt(0, 502);
		searchTerm.commitBuffer();
		
		search.search(keyValues, keyOrder, searchTerm, key_len, seq_len, false);
		assertEquals(search.getSearchPos(), 501);

		
		searchTerm.mapBuffer(Buffer.WRITE);
		searchTerm.writeInt(0, 0);
		searchTerm.commitBuffer();
		
		search.search(keyValues, keyOrder, searchTerm, key_len, seq_len, false);
		assertEquals(search.getSearchPos(), 0);
		
		for (int i = 0;i < seq_len; ++i)
		{
			values[i] = i * 2;
		}

		keyValues.mapBuffer(Buffer.WRITE);
		keyValues.writeArray(0,  values);
		keyValues.commitBuffer();
		
		searchTerm.mapBuffer(Buffer.WRITE);
		searchTerm.writeInt(0, 3);
		searchTerm.commitBuffer();

		 search.search(keyValues, keyOrder, searchTerm, key_len, seq_len, false);
		assertEquals(search.getSearchPos(), 1);
	}
	
	@Test
	public void compareSpeed()
	{
		int key_len = (int)DirectMemory.INT_SIZE;
		int seq_len = 10000000;
		
		int[] order = new int[seq_len];
		int[] values = new int[seq_len];
		for (int i = 0;i < seq_len; ++i)
		{
			order[i] = i;
			values[i] = i+1;
		}
		
		
		
		Context ctx = new Context( Context.DEFAULT_MEMORY, null);
		Buffer keyValues = new Buffer(ctx, seq_len* key_len);
		Buffer keyOrder = new Buffer(ctx, seq_len * DirectMemory.INT_SIZE);
		
		keyValues.mapBuffer(Buffer.WRITE);
		keyValues.writeArray(0,  values);
		keyValues.commitBuffer();
		
		keyOrder.mapBuffer(Buffer.WRITE);
		keyOrder.writeArray(0,  order);
		keyOrder.commitBuffer();
		
		Buffer searchTerm = new Buffer(ctx, key_len);
		searchTerm.mapBuffer(Buffer.WRITE);
		searchTerm.writeInt(0, 502);
		searchTerm.commitBuffer();
		
		
		NarySearch search = new NarySearch(ctx, false);
		
		search.search(keyValues, keyOrder, searchTerm, key_len, seq_len, false);
		
		
		long start = System.currentTimeMillis();
		for (int i = 0;i < 1000; ++i)
			search.search(keyValues, keyOrder, searchTerm, key_len, seq_len, true);
		ctx.finishDefaultQueue();
		long end = System.currentTimeMillis();
		System.out.println( "Sequence length = " + seq_len + " time "+ (end-start) + " msec/1000");
		int pos = 0;
		start = System.currentTimeMillis();
		for (int i = 0;i < 1000; ++i)
			pos = Arrays.binarySearch(values, 501);
		end = System.currentTimeMillis();
		System.out.println( "Sequence length = " + seq_len + " time "+ (end-start) + " msec/1000 " + pos);
	}
	
	
	

	public static void main(String[] args)
	{
		new SearchTest().testCreate();
	}

}

