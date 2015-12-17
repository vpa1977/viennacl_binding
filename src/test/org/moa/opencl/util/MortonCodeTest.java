package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import org.junit.Test;
import org.moa.opencl.util.CLogsSort;
import org.moa.opencl.util.MortonCode;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

public class MortonCodeTest {

	static
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
	@Test
	public void testCreate() {
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		int num_points = 10000;
		int num_dimensions = 2;
		MortonCode morton = new MortonCode(ctx, num_dimensions);
		
		byte[] data = new byte[num_dimensions* num_dimensions * 256];
		Buffer buf = morton.lookupTable();
		buf.mapBuffer(Buffer.READ);
		buf.readArray(0,data);
		buf.commitBuffer();
		
		
		CLogsSort sorter = new CLogsSort(ctx);
		
		Buffer output = new Buffer(ctx, num_points * num_dimensions * DirectMemory.INT_SIZE);
		Buffer output_cpu = new Buffer(ctx, num_points * num_dimensions * DirectMemory.INT_SIZE);
		int[] source_points = new int[num_dimensions*num_points];
		int[] morton_codes = new int[num_dimensions*num_points];
		int[] cpu_morton_codes = new int[num_dimensions*num_points];
		int offset = 0;
		for (int x = 0; x < 100; ++x)
			for (int y =0 ; y < 100; ++y )
			{
				source_points[offset++] = x;
				source_points[offset++] = y;
			}
		Buffer src = new Buffer(ctx, source_points.length * DirectMemory.INT_SIZE);
		src.mapBuffer(Buffer.WRITE);
		src.writeArray(0, source_points);
		src.commitBuffer();
		
		morton.computeMortonCode(output, src, num_points);
		
		data = new byte[num_dimensions*num_points *(int) DirectMemory.INT_SIZE];
		output.mapBuffer(Buffer.READ);
		output.readArray(0, data);
		output.commitBuffer();
//		for (int i = 0;i < data.length ; ++i)
//			System.out.println((int)data[i]);
		
//		morton.computeMortonCodeCPU(output_cpu, src, num_points);
//		output_cpu.mapBuffer(Buffer.READ);
//		output_cpu.readArray(0, data);
//		output_cpu.commitBuffer();
	//	for (int i = 0;i < data.length ; ++i)
	//		System.out.println((int)data[i]);	
		
		
	//	assertArrayEquals(cpu_morton_codes, morton_codes);
		
		Buffer key_indices =new Buffer(ctx, num_points* DirectMemory.INT_SIZE);
		Buffer value_indices =new Buffer(ctx, num_points* DirectMemory.INT_SIZE);
		sorter.sort(key_indices, output, value_indices, (int)(num_dimensions * DirectMemory.INT_SIZE), num_points);
		
		int[] result = new int[num_points];
		value_indices.mapBuffer(Buffer.READ);
		value_indices.readArray(0, result);
		value_indices.commitBuffer();
		
		for (int i = 0; i < result.length ; ++i)
		{
			for (int d = 0 ; d < num_dimensions ; ++d)
			{
				System.out.print( source_points[  result[i]* num_dimensions + d ] );
				System.out.print("\t");
			}
			System.out.println();
		}
		
		
	}

}
