package test.org.viennacl.binding.util;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Test;
import org.viennacl.binding.Context;

public class DoubleToFloatTest {

    public static void main(String[] args)
    {
        new DoubleToFloatTest().testConversionSpeed();
    }
    
	Context opencl_context = new Context(Context.Memory.OPENCL_MEMORY, null);
	Context cpu_context = new Context(Context.Memory.MAIN_MEMORY, null);

    
	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	/*
	@Test
	public void testSimpleConversion() {
		int len = 16384;
		Random rnd = new Random();
		DoubleToFloat cpu_converter = new DoubleToFloat(cpu_context, len);
		DoubleToFloat gpu_converter = new DoubleToFloat(opencl_context, len);
		double[] src = new double[len];
		float[] dst = new float[len];
		float[] dst_gpu = new float[len];
		for (int i = 0; i< len; ++i)
			src[i] = rnd.nextDouble();
		
		cpu_converter.convert(src,  dst);
		gpu_converter.convert(src, dst_gpu);
		
		assertArrayEquals(dst, dst_gpu, (float)0.0001);
		
	}
	*/
	
	@Test
	public void testConversionSpeed()
	{
		
		int len = 1000000;
		Random rnd = new Random();
		DoubleToFloat cpu_converter = new DoubleToFloat(cpu_context, len);
		DoubleToFloat gpu_converter = new DoubleToFloat(opencl_context, len);
		double[] src = new double[len];
		float[] dst = new float[len];
		float[] dst_gpu = new float[len];
		for (int i = 0; i< len; ++i)
			src[i] = rnd.nextDouble();
		gpu_converter.convert(src, dst_gpu);
		long cpu_start = System.nanoTime();
		cpu_converter.convert(src,  dst);
		long cpu_end = System.nanoTime();
		gpu_converter.convert(src, dst_gpu);
		long gpu_end = System.nanoTime();
		
		System.out.println( "cpu: "+ (cpu_end - cpu_start)  + "\ngpu: "+ (gpu_end - cpu_end));
		
		assertArrayEquals(dst, dst_gpu, (float)0.0001);
		
	}

}
