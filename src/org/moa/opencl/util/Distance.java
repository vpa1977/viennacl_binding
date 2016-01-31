package org.moa.opencl.util;

import org.moa.gpu.DenseInstanceBuffer;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.Kernel;

import weka.core.Instances;

public class Distance extends AbstractUtil {
	private Kernel m_square_distance_kernel;
	private Kernel m_square_distance_kernel_index;
	private Kernel m_s_t;

	public Distance(Context ctx)
	{
		if (!ctx.hasProgram("square_distance_double")) 
			init(ctx);
		m_square_distance_kernel = ctx.getKernel("square_distance_double", "square_distance");
		//m_square_distance_kernel_index= ctx.getKernel("square_distance_double", "square_distance_index");
		//m_s_t = ctx.getKernel("square_distance_double", "short_test");
	}

	private void init(Context ctx) {
		StringBuffer data = loadKernel("distance.cl");
		ctx.add("square_distance_double", "#define VALUE_TYPE double\n" + data.toString());
		
		
	}

	
	public void squareDistance(
			Instances dataset,
			DenseInstanceBuffer test_instance, 
			DenseInstanceBuffer instance_buffer, 
			Buffer min_buffer, 
			Buffer max_buffer, 
			Buffer attribute_types, 
			Buffer result, 
			int size, 
			Buffer indices
			) {
		int global_size = (int)size;
		m_square_distance_kernel.set_global_size(0, global_size);
		m_square_distance_kernel.set_arg(0, test_instance.attributes());
		m_square_distance_kernel.set_arg(1, instance_buffer.attributes());
		m_square_distance_kernel.set_arg(2, min_buffer);
		m_square_distance_kernel.set_arg(3, max_buffer);
		m_square_distance_kernel.set_arg(4, attribute_types);
		m_square_distance_kernel.set_arg(5, indices);
		m_square_distance_kernel.set_arg(6, result);
		m_square_distance_kernel.set_arg(7, dataset.numAttributes());
		m_square_distance_kernel.set_arg(8, 1);
		m_square_distance_kernel.invoke();
	}


	public void squareDistance(Instances dataset,
			DenseInstanceBuffer test_instance, 
			DenseInstanceBuffer instance_buffer, 
			Buffer min_buffer, 
			Buffer max_buffer, 
			Buffer attribute_types, 
			Buffer result 
			) {
		int global_size = (int)instance_buffer.rows();
		m_square_distance_kernel.set_global_size(0, global_size);
		m_square_distance_kernel.set_arg(0, test_instance.attributes());
		m_square_distance_kernel.set_arg(1, instance_buffer.attributes());
		m_square_distance_kernel.set_arg(2, min_buffer);
		m_square_distance_kernel.set_arg(3, max_buffer);
		m_square_distance_kernel.set_arg(4, attribute_types);
		m_square_distance_kernel.set_arg(5, result);
		m_square_distance_kernel.set_arg(6, result);
		m_square_distance_kernel.set_arg(7, dataset.numAttributes());
		m_square_distance_kernel.set_arg(8, 0);
		m_square_distance_kernel.invoke();
	}

}
