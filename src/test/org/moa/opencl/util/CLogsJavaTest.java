package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import org.junit.Test;
import org.moa.opencl.util.BufHelper;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.moa.opencl.util.Operations;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

public class CLogsJavaTest {
	
	static
	{
		System.loadLibrary("viennacl-java-binding");
	}

	
	public void testSortBinaryStrings() 
	{
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		CLogsVarKeyJava sort = new CLogsVarKeyJava(ctx, false);

		int size = 3;
		
		int key_size = 4;
		byte[] keys_data = {
				0,4,0,2,
				5,0,3,0,
				0,0,3,2,
				//0,0,0,1
		};
		
		//int key_size = 12;
		/*byte[] keys_data = 
			{ // 12 x 4
					0,0,0,0,0,0,0,0,1,0,0,1,
					0,0,0,0,0,0,0,0,0,0,1,0,
					0,0,0,0,0,0,0,0,0,0,0,1,
					0,0,0,0,0,1,0,0,0,0,1,0
			};*/

		Buffer key_values = new Buffer(ctx, 12 * size * DirectMemory.INT_SIZE);
		Buffer keys = new Buffer(ctx, size * DirectMemory.INT_SIZE);
		int[] indices = new int[]{0,1,2};
		
		keys.mapBuffer(Buffer.WRITE);
		keys.writeArray(0,  indices);
		keys.commitBuffer();
		
		key_values.mapBuffer(Buffer.WRITE);
		key_values.writeArray(0,  keys_data);
		key_values.commitBuffer();
		
		sort.sort(keys, key_values, null, key_size, size);
		
		int[] result = BufHelper.rbi(keys);
		assertArrayEquals(new int[]{2, 1, 0, 3} , result);
		
		
	}
	
	@Test
	public void testSortUINT() 
	{
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		CLogsVarKeyJava sort = new CLogsVarKeyJava(ctx, false);
		int size = 10000;
		Buffer values = new Buffer(ctx, size * DirectMemory.INT_SIZE);
		Buffer keys = new Buffer(ctx, size * DirectMemory.INT_SIZE);
		int[] sample = new int[size];
		for (int i = 0;i < sample.length; i++ )
			sample[i] = sample.length - i;
		keys.mapBuffer(Buffer.WRITE);
		keys.writeArray(0,  sample);
		keys.commitBuffer();
		sort.sortFixedBuffer(keys, values, size);
		
		keys.mapBuffer(Buffer.READ);
		keys.readArray(0,  sample);
		keys.commitBuffer();
		for (int i = 0;i < sample.length; i++ )
			assertEquals(sample[i], i+1);
    System.out.println("Sorted correctly");
	}
	
	
	public void testCreate() {
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		CLogsVarKeyJava sort = new CLogsVarKeyJava(ctx, true);
		Operations ops = new Operations(ctx);
		int size = 9999999;
		
		Buffer values = new Buffer(ctx, size * DirectMemory.INT_SIZE);
		Buffer keys = new Buffer(ctx, size * DirectMemory.INT_SIZE);
		Buffer key_values = new Buffer(ctx, size * DirectMemory.LONG_SIZE);
		long[] samples = new long[size];
		for (int i = 0 ; i < samples.length ; ++i)
			samples[i]= samples.length - i;
		
		key_values.mapBuffer(Buffer.WRITE);
		key_values.writeArray(0, samples);
		key_values.commitBuffer();
		
		ops.prepareOrderKey(keys, size);
		
		
		sort.sort(keys, key_values,values,(int) DirectMemory.LONG_SIZE, size);
		
		
		int[] sortIndices = new int[size];
		keys.mapBuffer(Buffer.READ);
		keys.readArray(0, sortIndices);
		keys.commitBuffer();
		int[] test = new int[size];
		for (int i = sortIndices.length-1; i>=0 ; --i)
			test[i]  = sortIndices.length -i -1;
		assertArrayEquals(test, sortIndices);
		// test with 32 bytes keys. 
		
		byte[] test_keys = 
			{ 1,2,3,4,5,6,7,8,9,0,11,12,13,14,15,16,17,18,19,20,21,
			  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
			  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
			  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
			  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,
			  0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
			};
			  
		key_values.mapBuffer(Buffer.WRITE);
		key_values.writeArray(0, test_keys);
		key_values.commitBuffer();
		ops.prepareOrderKey(keys, size);
		sort.sort(keys, key_values,values,(int) 21, 6);
		sortIndices = new int[6];
		keys.mapBuffer(Buffer.READ);
		keys.readArray(0, sortIndices);
		keys.commitBuffer();
		test = new int[6];
		for (int i = 0; i < sortIndices.length ; ++i)
			test[i]  = sortIndices.length - i - 1;
		assertArrayEquals(test, sortIndices);

	}
  
  public static void main(String[] arg)
  {
    System.out.println(System.getProperty("java.class.path"));
    new CLogsJavaTest().testSortBinaryStrings();
  }
	
	

}
