package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;

import org.junit.Test;
import org.moa.opencl.util.BufHelper;
import org.moa.opencl.util.CLogsSort;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.moa.opencl.util.MortonCode;
import org.moa.opencl.util.Operations;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.core.Instance;
import weka.core.NormalizableDistance;


public class MortonCodeTest {

	static
	{
		System.loadLibrary("viennacl-java-binding");
	}
  
  public static void main(String[] args) throws Throwable
  {
     new MortonCodeTest().testCreate2();
  }
  
  public BigInteger interleaveAsString(int[] point) throws Exception {
  	String[] binaryStringValues = new String[point.length];
  	int max_len = -1;
  	for (int i = 0; i < binaryStringValues.length ; ++i)
  	{
  		
  		binaryStringValues[i] = Integer.toBinaryString(point[i]);
  		if (binaryStringValues[i].length() > max_len)
  			max_len = binaryStringValues[i].length();
  	}
  	
  	for (int i = 0; i < binaryStringValues.length ; ++i)
  	{
  		while (binaryStringValues[i].length() < max_len)
  		{
  			binaryStringValues[i] = '0' +binaryStringValues[i]; 
  		}
  	}
  	String res = "";
  	for (int pos = 0; pos <max_len ; ++pos)
  	{
  		for (int i = binaryStringValues.length-1; i >=0 ; --i)
  		{
  			res += binaryStringValues[i].charAt(pos);
  		}
  		/*for (int i = 0; i<binaryStringValues.length ; ++i)
  		{
  			res += binaryStringValues[i].charAt(pos);
  		}*/
  	}
  	
  	BigInteger intValue= new BigInteger(res, 2);
  	
  	return intValue;
  }
  class Unit implements Comparable<Unit>
  {
	  BigInteger lol;
	  int[] point;
	  
	  public String toString() 
	  {
		  String res = "";
		  for (int i : point)
			  res = i + " " + res;
		  return res;
	  }
	@Override
	public int compareTo(Unit o) {
		// TODO Auto-generated method stub
		return lol.compareTo(o.lol);
	}
  }
  
  
  public ArrayList<Unit> createReferencePlot() throws Exception 
  {
	  int num_dimensions = 3;
	  int num_points = 10*10*10;
		int[] source_points = new int[num_dimensions*num_points];
		int[] morton_codes = new int[num_dimensions*num_points];
		int[] cpu_morton_codes = new int[num_dimensions*num_points];
		int offset = 0;
		ArrayList<Unit> seq = new ArrayList<Unit>();
		for (int x = 0; x < 10; ++x)
			for (int y =0 ; y < 10; ++y )
			{
				for (int z =0 ; z < 10; ++z )
				{
					int[] point = new int[]{ x, y, z };
					Unit d = new Unit();
					d.point = point;
					d.lol = interleaveAsString(d.point);
					seq.add(d);
				}
			}
		Collections.sort(seq);;
		
		for (Unit u : seq)
			System.out.println(u);
		
		return seq;
		
	  
  }
	
	@Test
	public void testCreate() {
		Context ctx = new Context(Context.DEFAULT_MEMORY, null);
		int num_points = 10000;
		int num_dimensions = 2;
		MortonCode morton = new MortonCode(ctx, num_dimensions);
		
		byte[] data = new byte[num_dimensions* num_dimensions * 256];
		Buffer buf = morton.lookupTable();
		buf.mapBuffer(Buffer.READ);
		buf.readArray(0,data);
		buf.commitBuffer();
		
		Operations ops = new Operations(ctx);
		CLogsVarKeyJava sorter = new CLogsVarKeyJava(ctx, false);
		
		Buffer output = new Buffer(ctx, num_points * num_dimensions * DirectMemory.INT_SIZE);
		Buffer output_cpu = new Buffer(ctx, num_points * num_dimensions * DirectMemory.INT_SIZE);
		int[] source_points = new int[num_dimensions*num_points];
		int[] morton_codes = new int[num_dimensions*num_points];
		int[] cpu_morton_codes = new int[num_dimensions*num_points];
		int offset = 0;
		for (int x = 0; x < 100; ++x)
			for (int y =0 ; y < 100; ++y )
			{
				source_points[offset++] = 100- x;
				source_points[offset++] = 100 -y;
			}
		Buffer src = new Buffer(ctx, source_points.length * DirectMemory.INT_SIZE);
		src.mapBuffer(Buffer.WRITE);
		src.writeArray(0, source_points);
		src.commitBuffer();
		
		morton.computeMortonCodeGroup(output, src, num_points);
	

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
		ops.prepareOrderKey(key_indices, num_points);
    
    int[] check = new int[num_points];
    key_indices.mapBuffer(Buffer.READ);
    key_indices.readArray(0, check);
    key_indices.commitBuffer();
    
		sorter.sort(key_indices, output, null, (int)(num_dimensions * DirectMemory.INT_SIZE), num_points);
		
		int[] result = new int[num_points];
		key_indices.mapBuffer(Buffer.READ);
		key_indices.readArray(0, result);
		key_indices.commitBuffer();
		
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
	  public static String byteToString(byte b) {
		    byte[] masks = { -128, 64, 32, 16, 8, 4, 2, 1 };
		    StringBuilder builder = new StringBuilder();
		    for (byte m : masks) {
		        if ((b & m) == m) {
		            builder.append('1');
		        } else {
		            builder.append('0');
		        }
		    }
		    return builder.toString();
		}
	@Test 
	public void generateGPUCurve() 
	{
		Context ctx = new Context(Context.DEFAULT_MEMORY, null);
		int num_points = 10*10*10;
		int num_dimensions = 3;
		MortonCode morton = new MortonCode(ctx, num_dimensions);
		
		byte[] data = new byte[num_dimensions* num_dimensions * 256];
		Buffer buf = morton.lookupTable();
		buf.mapBuffer(Buffer.READ);
		buf.readArray(0,data);
		buf.commitBuffer();
		
		Operations ops = new Operations(ctx);
		CLogsVarKeyJava sorter = new CLogsVarKeyJava(ctx, false);
		
		Buffer output = new Buffer(ctx, num_points * num_dimensions * DirectMemory.INT_SIZE);
		Buffer output_cpu = new Buffer(ctx, num_points * num_dimensions * DirectMemory.INT_SIZE);
		int[] source_points = new int[num_dimensions*num_points];
		int[] morton_codes = new int[num_dimensions*num_points];
		int[] cpu_morton_codes = new int[num_dimensions*num_points];
		int offset = 0;
		for (int x = 0; x < 10; ++x)
			for (int y =0 ; y < 10; ++y )
			{
				for (int z =0 ; z < 10; ++z )
				{
					source_points[offset++] = z;
					source_points[offset++] = y;
					source_points[offset++] = x;
				}
			}
		Buffer src = new Buffer(ctx, source_points.length * DirectMemory.INT_SIZE);
		src.mapBuffer(Buffer.WRITE);
		src.writeArray(0, source_points);
		src.commitBuffer();
		
		morton.computeMortonCode(output, src, num_points);
		
		Buffer key_indices =new Buffer(ctx, num_points* DirectMemory.INT_SIZE);
		Buffer value_indices =new Buffer(ctx, num_points* DirectMemory.INT_SIZE);
		ops.prepareOrderKey(key_indices, num_points);
    
    int[] check = new int[num_points];
    key_indices.mapBuffer(Buffer.READ);
    key_indices.readArray(0, check);
    key_indices.commitBuffer();
    
		sorter.sort(key_indices, output, null, (int)(num_dimensions * DirectMemory.INT_SIZE), num_points);
		
		int[] result = new int[num_points];
		key_indices.mapBuffer(Buffer.READ);
		key_indices.readArray(0, result);
		key_indices.commitBuffer();
		
		
		for (int i = 0; i < result.length; i+=3)
		{
			int off = result[i]* num_dimensions;
			System.out.println(source_points[off] + "\t"+source_points[off+1]+"\t"+source_points[off+2]);
		}
		
	}
	
	@Test
	public void testCreate2() throws Exception {
		Context ctx = new Context(Context.DEFAULT_MEMORY, null);
		int num_points = 10*10*5;
		int num_dimensions = 3;
		MortonCode morton = new MortonCode(ctx, num_dimensions);
		
		byte[] data = new byte[num_dimensions* num_dimensions * 256];
		Buffer buf = morton.lookupTable();
		buf.mapBuffer(Buffer.READ);
		buf.readArray(0,data);
		buf.commitBuffer();
		
		Operations ops = new Operations(ctx);
		CLogsVarKeyJava sorter = new CLogsVarKeyJava(ctx, false);
		
		Buffer output = new Buffer(ctx, num_points * num_dimensions * DirectMemory.INT_SIZE);
		Buffer output_cpu = new Buffer(ctx, num_points * num_dimensions * DirectMemory.INT_SIZE);
		int[] source_points = new int[num_dimensions*num_points];
		int[] morton_codes = new int[num_dimensions*num_points];
		int[] cpu_morton_codes = new int[num_dimensions*num_points];
		int offset = 0;
		for (int x = 0; x < 10; ++x)
			for (int y =0 ; y < 10; ++y )
			{
				for (int z =0 ; z < 5; ++z )
				{
					source_points[offset++] = x;
					source_points[offset++] = y;
					source_points[offset++] = z;
				}
			}
		Buffer src = new Buffer(ctx, source_points.length * DirectMemory.INT_SIZE);
		src.mapBuffer(Buffer.WRITE);
		src.writeArray(0, source_points);
		src.commitBuffer();
		
		morton.computeMortonCode(output, src, num_points);
		byte[] codes = BufHelper.rbb(output);
		
		for (int i = 0; i < source_points.length; i+=3)
		{
			System.out.print(source_points[i] + ","+source_points[i+1]+","+source_points[i+2]+"=");
			int code_off = (int)((i/3) * num_dimensions * DirectMemory.INT_SIZE);
			int code_len = (int)(num_dimensions * DirectMemory.INT_SIZE);
			for (int j = code_off; j < code_off  +code_len; ++j)
				System.out.print(byteToString(codes[j]));
			System.out.println();
		}

		
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
		ops.prepareOrderKey(key_indices, num_points);
    
    int[] check = new int[num_points];
    key_indices.mapBuffer(Buffer.READ);
    key_indices.readArray(0, check);
    key_indices.commitBuffer();
    
		sorter.sort(key_indices, output, null, (int)(num_dimensions * DirectMemory.INT_SIZE), num_points);
		
		int[] result = new int[num_points];
		key_indices.mapBuffer(Buffer.READ);
		key_indices.readArray(0, result);
		key_indices.commitBuffer();
		
		ArrayList<Unit> arr = createReferencePlot();
		
		
		for (int i = 0; i < result.length; i+=3)
		{
			int off = result[i]* num_dimensions;
			System.out.print(source_points[off] + ","+source_points[off+1]+","+source_points[off+2]+"=");
			int code_off = (int)((off/num_dimensions) * num_dimensions * DirectMemory.INT_SIZE);
			int code_len = (int)(num_dimensions * DirectMemory.INT_SIZE);
			for (int j = code_off; j < code_off  +code_len; ++j)
				System.out.print(byteToString(codes[j]));
			System.out.println();
		}

		
		for (int i = 0; i < result.length ; ++i)
		{
			String aCode = "";
			int[] ref_point = new int[num_dimensions];
			for (int d = 0 ; d < num_dimensions ; ++d)
			{
				ref_point[d]  = source_points[  result[i]* num_dimensions + d ];
			}
			for (int pos = result[i]*num_dimensions*num_dimensions; pos <result[i]*num_dimensions*num_dimensions+num_dimensions; ++pos  )
				aCode += byteToString(codes[ pos ]);
			
			BigInteger bi = interleaveAsString(ref_point);
			String refString = bi.toString(2);
			System.out.println(aCode);
			System.out.println(refString);
			
			Unit u = arr.get(i);
			for (int d = 0 ; d < num_dimensions ; ++d)
			{
//				assertEquals(u.point[d],source_points[  result[i]* num_dimensions + d ] );
				System.out.print( source_points[  result[i]* num_dimensions + d ] );
				
				System.out.print("\t");
			}
			System.out.println();
		}
		
		
	}

 
}
