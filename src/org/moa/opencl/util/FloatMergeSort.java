package org.moa.opencl.util;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

public class FloatMergeSort {
	private Buffer m_temp;
	private Buffer m_dst;
	private Context m_context;

	public FloatMergeSort(Context ctx, int bufferLength )
	{
		m_temp = new Buffer(ctx, bufferLength * DirectMemory.DOUBLE_SIZE, Buffer.WRITE);
		m_dst = new Buffer(ctx, bufferLength * DirectMemory.INT_SIZE, Buffer.WRITE);
		m_context = ctx;
		init(m_context);
	}
	

	public void  sort(Buffer input, Buffer indices)
	{
		nativeSort(input, m_temp, indices, m_dst);
	}

	@Override
	protected void finalize() throws Throwable {
		release();
		// TODO Auto-generated method stub
		super.finalize();
		
	}
	
	private native void init(Context ctx);
	private native void release();
	private native void nativeSort(Buffer in, Buffer temp, Buffer src, Buffer dst);
	private long m_native_context;
}
