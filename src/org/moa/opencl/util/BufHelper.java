package org.moa.opencl.util;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

public class BufHelper {
	public static double[] rb(Buffer b)
	{
		double[] ret = new double[(int)(b.byteSize()/DirectMemory.DOUBLE_SIZE)];
		b.mapBuffer(Buffer.READ);
		b.readArray(0,  ret);
		b.commitBuffer();
		return ret;
	}
	
	public static int[] rbi(Buffer b) 
	{
		int[] ret = new int[(int)(b.byteSize()/DirectMemory.INT_SIZE)];
		b.mapBuffer(Buffer.READ);
		b.readArray(0,  ret);
		b.commitBuffer();
		return ret;
	}
	
	public static Buffer wb(Context ctx, double[] data)
	{
		Buffer b = new Buffer(ctx, data.length * DirectMemory.DOUBLE_SIZE);
		b.mapBuffer(Buffer.WRITE);
		b.writeArray(0,  data);
		b.commitBuffer();
		return b;
	}

	public static float[] rbf(Buffer b) {
		float[] ret = new float[(int)(b.byteSize()/DirectMemory.FLOAT_SIZE)];
		b.mapBuffer(Buffer.READ);
		b.readArray(0,  ret);
		b.commitBuffer();
		return ret;
	}

}
