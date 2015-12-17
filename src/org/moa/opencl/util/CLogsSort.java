package org.moa.opencl.util;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.Kernel;

/** 
 * Interface to the CLOG sorting routine
 * @author john
 *
 */
public class CLogsSort extends AbstractUtil{
	private Kernel m_fill_kernel;
	public CLogsSort(Context context)
	{
		m_context = context;
		init();
		if (!context.hasProgram("clogs_support"))
		{
			StringBuffer code = loadKernel("clogs_prepare_buffer.cl");
			context.add("clogs_support", code.toString());
		}
		m_fill_kernel = context.getKernel("clogs_support", "prepare_buffer");
	}
	
	public void prepareOrderKey(Buffer order_key, int size)
	{
		m_fill_kernel.set_global_size(0, size);
		m_fill_kernel.set_arg(0, order_key);
		m_fill_kernel.invoke();
	}
	/**
	 * 
	 * @param order_key - temporary buffer containing key identifiers 0.... N
	 * @param key_values - actual key values (key_length bytes)
	 * @param values - output with the sort order of key_values
	 * @param key_length - key length in bytes
	 * @param size - number of keys
	 */
	public void sort(Buffer order_key, Buffer key_values,Buffer values, int key_length, int size)
	{
		prepareOrderKey(order_key, size);
		order_key.copyTo(values);
		nativeSort(order_key, key_values, values, key_length, size);
	}
	
	private native void nativeSort(Buffer order_key, Buffer key_values, Buffer values, int key_length, int size);

	@Override
	protected void finalize() throws Throwable {
		release();
		super.finalize();
	}



	public native void release();
	private native void init();

	
	private Context m_context;
	private long m_native_context;
}
