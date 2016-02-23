package org.moa.opencl.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.moa.gpu.DenseInstanceBuffer;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

import weka.core.Instances;

public class MinMax extends AbstractUtil {
	
	private Kernel m_full_min_max_kernel;
	private Kernel m_update_min_max_kernel;
	private Kernel m_full_min_max_kernel_float;
	private Kernel m_update_min_max_kernel_float;

	private Kernel m_full_min_max_kernel_float_indices;
	private Kernel m_update_min_max_kernel_float_indices;

	public MinMax(Context ctx)
	{
		if (!ctx.hasProgram("min_max_double")) 
			init(ctx);
		m_full_min_max_kernel = ctx.getKernel("min_max_double", "min_max_kernel");
		m_update_min_max_kernel = ctx.getKernel("min_max_double", "min_max_update_kernel");
		
		m_full_min_max_kernel_float = ctx.getKernel("min_max_float", "min_max_kernel");
		m_update_min_max_kernel_float = ctx.getKernel("min_max_float", "min_max_update_kernel");

		m_full_min_max_kernel_float_indices = ctx.getKernel("min_max_float", "min_max_kernel_indices");
		

	}

	private void init(Context ctx) {
		StringBuffer data = loadKernel("minmax.cl");
		ctx.add("min_max_double", "#define VALUE_TYPE double\n" + data.toString());
		ctx.add("min_max_float",  "#define VALUE_TYPE float\n" + data.toString());
	}


	public void fullMinMaxDouble(Instances dataset, DenseInstanceBuffer instance_buffer, Buffer min_buffer, Buffer max_buffer) {
		int global_size = (int)(dataset.numAttributes())*256;
		m_full_min_max_kernel.set_global_size(0, global_size);
		m_full_min_max_kernel.set_local_size(0, 256);
		m_full_min_max_kernel.set_arg(0, dataset.classIndex());
		m_full_min_max_kernel.set_arg(1, dataset.numAttributes());
		m_full_min_max_kernel.set_arg(2, instance_buffer.rows());
		m_full_min_max_kernel.set_arg(3,  instance_buffer.attributes());
		m_full_min_max_kernel.set_arg(4,  min_buffer);
		m_full_min_max_kernel.set_arg(5,  max_buffer);
		m_full_min_max_kernel.invoke();
	}
	
	public void updateMinMaxDouble(Instances dataset, DenseInstanceBuffer instance_buffer, Buffer min_buffer, Buffer max_buffer) {
		int global_size = (int)dataset.numAttributes();
		m_update_min_max_kernel.set_global_size(0, global_size);
		m_update_min_max_kernel.set_local_size(0, 256);
		m_update_min_max_kernel.set_arg(0, dataset.classIndex());
		m_update_min_max_kernel.set_arg(1, dataset.numAttributes());
		m_update_min_max_kernel.set_arg(2,  instance_buffer.attributes());
		m_update_min_max_kernel.set_arg(3,  min_buffer);
		m_update_min_max_kernel.set_arg(4,  max_buffer);
		m_update_min_max_kernel.invoke();
	}
	
	
	public void fullMinMaxFloat(Instances dataset, DenseInstanceBuffer instance_buffer, Buffer min_buffer, Buffer max_buffer) {
		int global_size = (int)(dataset.numAttributes())*256;
		m_full_min_max_kernel_float.set_global_size(0, global_size);
		m_full_min_max_kernel_float.set_local_size(0, 256);
		m_full_min_max_kernel_float.set_arg(0, dataset.classIndex());
		m_full_min_max_kernel_float.set_arg(1, dataset.numAttributes());
		m_full_min_max_kernel_float.set_arg(2, instance_buffer.rows());
		m_full_min_max_kernel_float.set_arg(3,  instance_buffer.attributes());
		m_full_min_max_kernel_float.set_arg(4,  min_buffer);
		m_full_min_max_kernel_float.set_arg(5,  max_buffer);
		m_full_min_max_kernel_float.invoke();
	}
	
	public void fullMinMaxFloatIndices(Instances dataset, 
          DenseInstanceBuffer instance_buffer, Buffer min_buffer, Buffer max_buffer, Buffer indices, int length) {
		int global_size = (int)(dataset.numAttributes())*256;
		m_full_min_max_kernel_float_indices.set_global_size(0, global_size);
		m_full_min_max_kernel_float_indices.set_local_size(0, 256);
		m_full_min_max_kernel_float_indices.set_arg(0,  indices);
		m_full_min_max_kernel_float_indices.set_arg(1, dataset.classIndex());
		m_full_min_max_kernel_float_indices.set_arg(2, dataset.numAttributes());
		m_full_min_max_kernel_float_indices.set_arg(3, length);
		m_full_min_max_kernel_float_indices.set_arg(4,  instance_buffer.attributes());
		m_full_min_max_kernel_float_indices.set_arg(5,  min_buffer);
		m_full_min_max_kernel_float_indices.set_arg(6,  max_buffer);
		m_full_min_max_kernel_float_indices.invoke();
	}

	
	public void updateMinMaxFloat(Instances dataset, DenseInstanceBuffer instance_buffer, Buffer min_buffer, Buffer max_buffer) {
		int global_size = (int)dataset.numAttributes();
		m_update_min_max_kernel_float.set_global_size(0, global_size);
		m_update_min_max_kernel_float.set_local_size(0, 256);
		m_update_min_max_kernel_float.set_arg(0, dataset.classIndex());
		m_update_min_max_kernel_float.set_arg(1, dataset.numAttributes());
		m_update_min_max_kernel_float.set_arg(2,  instance_buffer.attributes());
		m_update_min_max_kernel_float.set_arg(3,  min_buffer);
		m_update_min_max_kernel_float.set_arg(4,  max_buffer);
		m_update_min_max_kernel_float.invoke();
	}

}
