package org.moa.gpu;

import java.util.Random;
import java.util.stream.DoubleStream;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

/** 
 * Implements a fast johnson lidenstrauss transform  
 * N is expected to be power of 2 for fast FFT.
 *
 */
public class FJLT {
	private double m_srhst_constant;
	private String DATA_TYPE = "double";
	private int m_n;
	private int m_k;
	private double m_scale;
	private Buffer m_substitutions_buffer;
	private Buffer m_indicators_buffer;
	private Buffer m_transform_temp;
	private Buffer m_pre_fft_temp;
	private Kernel m_permute_kernel;
	public FJLT(Context ctx, int n, int k)
	{
		m_n = n;
		m_k = k;
		m_scale = 1/ Math.sqrt(2 * m_n);

		m_srhst_constant = Math.sqrt(n/k);
		m_substitutions_buffer = new Buffer(ctx, k * DirectMemory.INT_SIZE);
		m_indicators_buffer = new Buffer(ctx, n * DirectMemory.DOUBLE_SIZE);
		m_transform_temp = new Buffer(ctx, n* DirectMemory.DOUBLE_SIZE);
		m_pre_fft_temp = new Buffer(ctx, n* DirectMemory.DOUBLE_SIZE);
		int[]  substitutions = new int[k];
		double[]  indicators = new double[n];
		Random rnd = new Random();
		for (int i = 0; i < indicators.length; ++i)
		{
			do {
				indicators[i] = m_scale * Math.signum(rnd.nextDouble()-0.5);
			} while (indicators[i] ==0);
		}
		for (int i = 0; i< k ; ++i)
			substitutions[i] = rnd.nextInt(n);
		m_substitutions_buffer.mapBuffer(Buffer.WRITE);
		m_indicators_buffer.mapBuffer(Buffer.WRITE);
		m_substitutions_buffer.writeArray(0, substitutions);
		m_indicators_buffer.writeArray(0, indicators);
		
		m_substitutions_buffer.commitBuffer();
		m_indicators_buffer.commitBuffer();
		
		if (ctx.memoryType() != ctx.MAIN_MEMORY)
		{
			if (!ctx.hasProgram("fjlt") )
				initKernels(ctx);
			m_permute_kernel = ctx.getKernel("fjlt", "permute");
		}
		else
		{
			m_permute_kernel = null;
		}
	}
	
	private void initKernels(Context ctx)
	{
		String permute = 
			"__kernel void permute(__global const " + DATA_TYPE + "* src, __global const int* perm, "+DATA_TYPE+" srhst_const, __global " + DATA_TYPE + "* out)\n"+
			"{"+
			 "out[get_global_id(0)] = srhst_const * src[ perm[get_global_id(0)]];"+
			"};\n";
		
		ctx.add("fjlt", permute);
	}
	
	/** 
	 * Perform a FJLT transform
	 * @param source - source vector (all nominal are mapped to binary), size N 
	 * @param dest - destination vector size K. 
	 */
	public void transform(Buffer source, Buffer dest)
	{
		if (source.byteSize()/DirectMemory.DOUBLE_SIZE != m_n)
			throw new RuntimeException("Invalid source size");
		if (dest.byteSize()/ DirectMemory.DOUBLE_SIZE != m_k)
			throw new RuntimeException("Invalid destination size");
		native_transform(source,  m_indicators_buffer,  m_n, m_pre_fft_temp, m_transform_temp);
		if (m_permute_kernel!= null)
		{
			m_permute_kernel.set_arg(0, m_transform_temp);
			m_permute_kernel.set_arg(1, m_substitutions_buffer);
			m_permute_kernel.set_arg(2, m_srhst_constant);
			m_permute_kernel.set_arg(3, dest);
		}
		else
		{
			m_substitutions_buffer.mapBuffer(Buffer.READ);
			source.mapBuffer(Buffer.READ);
			dest.mapBuffer(Buffer.WRITE);
			m_transform_temp.mapBuffer(Buffer.READ);
			for (int i = 0; i < m_k; ++i)
			{
				int offset = m_substitutions_buffer.readInt(i*DirectMemory.INT_SIZE);
				double val = m_transform_temp.read(offset*DirectMemory.DOUBLE_SIZE);
				dest.write(i*DirectMemory.DOUBLE_SIZE, val * m_srhst_constant);
			}
			source.commitBuffer();
			dest.commitBuffer();
			m_transform_temp.commitBuffer();
			m_substitutions_buffer.commitBuffer();
		}
	}

	private native void native_transform(Buffer input, Buffer indicators,  int n, Buffer temp,Buffer pre_fft_temp );

}

