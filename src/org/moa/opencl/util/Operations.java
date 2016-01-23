package org.moa.opencl.util;

import java.util.jar.Attributes;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

public class Operations extends AbstractUtil {
	private Context m_context;
	private Kernel m_normalize_kernel;
	private Kernel m_double2uint_kernel;
	private Kernel m_prepare_order_key;
	public Operations(Context ctx)
	{
		m_context = ctx;
		if (!m_context.hasProgram("operations"))
			init(m_context);
		m_normalize_kernel = m_context.getKernel("operations", "normalize_attributes");
		m_double2uint_kernel = m_context.getKernel("operations", "double2uint");
		m_prepare_order_key = m_context.getKernel("operations", "prepare_order_key");
	}
	
	
	public void prepareOrderKey(Buffer order_key, int size)
	{
		m_prepare_order_key.set_global_size(0, size);
		m_prepare_order_key.set_arg(0, order_key);
		m_prepare_order_key.invoke();
	}
	public void doubleToInt32(Buffer double_buffer, Buffer int32_buffer, int rows, int num_attributes)
	{
		m_double2uint_kernel.set_global_size(0,rows * num_attributes);
		m_double2uint_kernel.set_arg(0, double_buffer);
		m_double2uint_kernel.set_arg(1, int32_buffer);
		m_double2uint_kernel.set_arg(2, (int)1000000);
		m_double2uint_kernel.invoke();
		
	}
	
	public void normalize(Buffer input, 
			Buffer output, 
			Buffer min_values, 
			Buffer max_values, 
			Buffer attribute_map, 
			int num_attributes, int num_instances)
	{
		m_normalize_kernel.set_global_size(0, num_instances);
		m_normalize_kernel.set_arg(0,input);
		m_normalize_kernel.set_arg(1,output);
		m_normalize_kernel.set_arg(2, min_values);
		m_normalize_kernel.set_arg(3, max_values);
		m_normalize_kernel.set_arg(4, attribute_map);
		m_normalize_kernel.set_arg(5, num_attributes);
		m_normalize_kernel.invoke();

	}
	
	private void init(Context ctx) {
		StringBuffer code = loadKernel("operations.cl");
		ctx.add("operations", code.toString());
		
	}
	
}
