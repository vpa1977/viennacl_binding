package org.moa.gpu;

import java.util.Random;

import org.moa.opencl.util.BufHelper;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.filters.unsupervised.attribute.RandomProjection;

public class MatrixRandomProjection {
	
	private RandomProjectionMatrixGenerator m_generator;
	private Buffer proj_buffer;
	private int m_k;
	private int m_n;
	
	private  class RandomProjectionMatrixGenerator extends RandomProjection
	{
		
		public float[] generate(int k, int n) 
		{
			m_random = new Random();
			m_random.setSeed(m_rndmSeed);

			float[] proj = new float[k*n];
			for(int i=0; i<k; i++) 
			     for(int j=0; j<n; j++) 
			    	  proj[i*n +j] =(float) rndmNum(true);
			return proj;

		}

		public float[] softProject(float[] data) {
			float[] out = new float[m_k];
			float[] proj = BufHelper.rbf(proj_buffer);
			int off = 0; 
			
			for (int i = 0 ; i <MatrixRandomProjection.this.m_k ; ++i)
				for(int j = 0; j < MatrixRandomProjection.this.m_n; ++j)
					out[i]+=  proj[off++] * data[j];
			return out;
		}
	}

	public MatrixRandomProjection(Context ctx, int k, int n)
	{
		m_generator = new RandomProjectionMatrixGenerator();
		float[] rand_matrix = m_generator.generate(k, n);
		proj_buffer = new Buffer(ctx, k*n*DirectMemory.FLOAT_SIZE);
		proj_buffer.mapBuffer(Buffer.WRITE);
		proj_buffer.writeArray(0,rand_matrix);
		proj_buffer.commitBuffer();
		m_k = k;
		m_n = n;
	}
	
	public void project(Buffer input, int rows, Buffer  output)
	{
		project(output, rows, input, proj_buffer, m_k, m_n);
	}
	
	public void project(Buffer input , Buffer output)
	{
		project(output,  input, proj_buffer, m_k, m_n);
	}

	
	// matrix-matrix product
	private native void project(Buffer output, int rows, Buffer input, Buffer proj, int k, int n);
	
	private native void project(Buffer output, Buffer input, Buffer proj, int k, int n);

	public float[] softProject(float[] data) {
		return m_generator.softProject(data);
	}
}
