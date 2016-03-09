package org.moa.gpu;

import java.util.Random;
import java.util.stream.DoubleStream;

import org.moa.opencl.util.AbstractUtil;
import org.moa.opencl.util.TreeUtil;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

import weka.filters.unsupervised.attribute.RandomProjection;

/** 
 * Implements a fast johnson lidenstrauss transform  
 * N is expected to be power of 2 for fast FFT.
 *
 */
public class FloatFJLT extends AbstractUtil{
	private Context m_context;
	private double m_srhst_constant;
	private String DATA_TYPE = "float";
	private int m_n;
	private int m_k;
	private double m_scale;
	private Buffer m_substitutions_buffer;
	private Buffer m_indicators_buffer;
	private Buffer m_transform_temp;
	private Buffer m_pre_fft_temp;
	private Buffer m_pow2_src;
	private Kernel m_permute_kernel;
	private Kernel m_fix_nan_kernel;
	
	
	public FloatFJLT(Context ctx, int k, int n)
	{
		m_context = ctx;
		m_n = n;
		m_k = k;
		m_scale = 1/ Math.sqrt(2 * m_n);
		int next_p2 = (int)TreeUtil.nextPow2(m_n);
		m_pow2_src = new Buffer(ctx, next_p2 * DirectMemory.FLOAT_SIZE);
		m_pow2_src.fill((byte)0);
		m_srhst_constant = Math.sqrt(n/k);
		m_substitutions_buffer = new Buffer(ctx, k * DirectMemory.INT_SIZE);
		m_indicators_buffer = new Buffer(ctx, next_p2 * DirectMemory.FLOAT_SIZE);
		m_transform_temp = new Buffer(ctx, next_p2* DirectMemory.FLOAT_SIZE);
		m_pre_fft_temp = new Buffer(ctx, next_p2* DirectMemory.FLOAT_SIZE);
		int[]  substitutions = new int[k];
		float[]  indicators = new float[next_p2];
		Random rnd = new Random();
		for (int i = 0; i < indicators.length; ++i)
		{
			do {
				indicators[i] =(float)( m_scale * Math.signum(rnd.nextDouble()-0.5));
			} while (indicators[i] ==0);
		}
		for (int i = 0; i< k ; ++i)
		{
			substitutions[i] = rnd.nextInt(n);
			for (int j = 0; j < i; ++j)
				if (substitutions[j] == substitutions[i])
				{
					--i;
					break;
				}
		}
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
		StringBuffer temp = new StringBuffer();
		temp.append("#define DATA_TYPE " + DATA_TYPE + "\n");
		temp.append(loadKernel("fjlt_float.cl"));
		
		ctx.add("fjlt", temp.toString());
	}
	
	/** 
	 * Perform a FJLT transform
	 * @param source - source vector (all nominal are mapped to binary), size N 
	 * @param dest - destination vector size K. 
	 */
	public void transform(Buffer source, Buffer dest)
	{
		if (source.byteSize()/DirectMemory.FLOAT_SIZE < m_n)
			throw new RuntimeException("Invalid source size");
		if (dest.byteSize()/ DirectMemory.FLOAT_SIZE < m_k)
			throw new RuntimeException("Invalid destination size");
		fix_nan(source,m_n);
		
		source.copyTo(m_pow2_src);
		native_transform(m_pow2_src,  m_indicators_buffer, (int) TreeUtil.nextPow2(m_n), m_pre_fft_temp, m_transform_temp);
		if (m_permute_kernel!= null)
		{
			m_permute_kernel.set_arg(0, m_k);
			m_permute_kernel.set_arg(1, m_transform_temp);
			m_permute_kernel.set_arg(2, m_substitutions_buffer);
			m_permute_kernel.set_arg(3, m_srhst_constant);
			m_permute_kernel.set_arg(4, dest);
			m_permute_kernel.set_local_size(0, 256);
			m_permute_kernel.set_global_size(0, 40 * 256);
			m_permute_kernel.invoke();
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
  
  public void transform(Buffer source, int rows, Buffer dst)
  {
	  fix_nan(source,m_n*rows);
	  native_batch_update(m_k,m_n,source, rows, m_pow2_src, m_transform_temp, m_substitutions_buffer,m_srhst_constant, dst, m_indicators_buffer );
  }
  

	private native void native_batch_update(int k, int n,Buffer source, int rows, Buffer pow2_src, Buffer transform_temp,
		Buffer substitutions_buffer, double srhst_constant, 
		Buffer dst, Buffer indicators);

	private void fix_nan(Buffer source, int size) {
	/*	if (m_fix_nan_kernel == null )
		{
			double[] test = new double[(int)(source.byteSize()/DirectMemory.DOUBLE_SIZE)];
			source.mapBuffer(Buffer.READ_WRITE);
			source.readArray(0, test);
			for (int i = 0;i < test.length ; ++i)
			{
				if (Double.isNaN(test[i]))
					test[i] = 0;
			}
			source.writeArray(0, test);
			source.commitBuffer();
		}
		else
		{
			m_fix_nan_kernel.set_global_size(0, 256 * 40);
			m_fix_nan_kernel.set_local_size(0, 256);
			m_fix_nan_kernel.set_arg(0, size);
			m_fix_nan_kernel.set_arg(1, source);
			m_fix_nan_kernel.invoke();
		}*/
		
	}

	private native void native_transform(Buffer input, Buffer indicators,  int n, Buffer temp,Buffer pre_fft_temp );

}

