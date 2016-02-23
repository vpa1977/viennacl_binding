package org.moa.opencl.util;

import java.util.jar.Attributes;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

public class Operations extends AbstractUtil {
	private static final int WG_COUNT = 40;
	private Context m_context;
	private Kernel m_normalize_kernel;
  private Kernel m_normalize_kernel_float;
	private Kernel m_double2uint_kernel;
	private Kernel m_prepare_order_key;
	private Kernel m_random_shift_kernel;
	private Kernel m_binary_search_kernel;

	public Operations(Context ctx) {
		m_context = ctx;
		if (m_context.memoryType() != Context.MAIN_MEMORY)
		{
			if (!m_context.hasProgram("operations"))
				init(m_context);
      m_normalize_kernel_float = m_context.getKernel("operations", "normalize_attributes_float");
			m_normalize_kernel = m_context.getKernel("operations", "normalize_attributes");
			m_double2uint_kernel = m_context.getKernel("operations", "double2uint");
			m_prepare_order_key = m_context.getKernel("operations", "prepare_order_key");
			m_random_shift_kernel = m_context.getKernel("operations", "random_shift");
		}
		// m_binary_search_kernel = m_context.getKernel("operations",
		// "binary_search");
	}
	
	

	public void prepareOrderKey(Buffer order_key, int size) {
		m_prepare_order_key.set_global_size(0, 128*WG_COUNT);
		m_prepare_order_key.set_local_size(0, 128);
		m_prepare_order_key.set_arg(0, order_key);
		m_prepare_order_key.set_arg(1, size);
		m_prepare_order_key.invoke();
	}

	public void doubleToInt32(Buffer double_buffer, Buffer attribute_map, Buffer int32_buffer, int rows,
			int num_attributes) {
		m_double2uint_kernel.set_global_size(0, 128*WG_COUNT);
		m_double2uint_kernel.set_local_size(0, 128);
		m_double2uint_kernel.set_arg(0, double_buffer);
		m_double2uint_kernel.set_arg(1, attribute_map);
		m_double2uint_kernel.set_arg(2, int32_buffer);
		m_double2uint_kernel.set_arg(3, (int) 1000000);
		m_double2uint_kernel.set_arg(4, rows * num_attributes);
		m_double2uint_kernel.set_arg(5, num_attributes);
		m_double2uint_kernel.invoke();

	}

	public void normalize(Buffer input, Buffer output, Buffer min_values, Buffer max_values, Buffer attribute_map,
			int num_attributes, int num_instances) {
		int size = num_instances;
		m_normalize_kernel.set_global_size(0, 128*(size/128+1));
		m_normalize_kernel.set_local_size(0, 128);
		m_normalize_kernel.set_arg(0, input);
		m_normalize_kernel.set_arg(1, output);
		m_normalize_kernel.set_arg(2, min_values);
		m_normalize_kernel.set_arg(3, max_values);
		m_normalize_kernel.set_arg(4, attribute_map);
		m_normalize_kernel.set_arg(5, num_attributes);
		m_normalize_kernel.set_arg(6, num_instances);
		m_normalize_kernel.invoke();

	}
  
  public void normalizeFloat(Buffer input, Buffer output, Buffer min_values, Buffer max_values, Buffer attribute_map,
			int num_attributes, int num_instances) {
	  int size = num_instances;
		m_normalize_kernel_float.set_global_size(0, 256 * WG_COUNT);
		m_normalize_kernel_float.set_local_size(0, 256);
		m_normalize_kernel_float.set_arg(0, input);
		m_normalize_kernel_float.set_arg(1, output);
		m_normalize_kernel_float.set_arg(2, min_values);
		m_normalize_kernel_float.set_arg(3, max_values);
		m_normalize_kernel_float.set_arg(4, attribute_map);
		m_normalize_kernel_float.set_arg(5, num_attributes);
		m_normalize_kernel_float.set_arg(6, num_instances);
		m_normalize_kernel_float.invoke();

	}

	private void init(Context ctx) {
		StringBuffer code = loadKernel("operations.cl");
		ctx.add("operations", code.toString());

	}

	public void shiftByRandomVector(Buffer data_point_buffer, Buffer random_shift, int num_attributes, int num_rows) {
		m_random_shift_kernel.set_global_size(0, num_attributes);
		m_random_shift_kernel.set_local_size(0, 1);
		int size = num_rows;
		m_random_shift_kernel.set_global_size(1, 128*WG_COUNT);
		m_random_shift_kernel.set_local_size(1, 128);
		m_random_shift_kernel.set_arg(0, data_point_buffer);
		m_random_shift_kernel.set_arg(1, random_shift);
		m_random_shift_kernel.set_arg(2, num_rows);
		m_random_shift_kernel.invoke();
	}

	
	
	public native void dense_ax(Buffer matrix, Buffer vector, Buffer output, int rows, int columns);

}
