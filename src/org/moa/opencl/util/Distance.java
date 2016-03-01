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
	private Kernel m_cosine_distance_kernel;
	private Kernel m_cosine_distance_norm_kernel;
	private Kernel m_square_distance_kernel_float;
  
  private int WG_COUNT = 4;
  private int WG_SIZE = 128;
  
	public Distance(Context ctx)
	{
		if (!ctx.hasProgram("square_distance_double")) 
			init(ctx);
		m_square_distance_kernel = ctx.getKernel("square_distance_double", "square_distance");
		m_square_distance_kernel_float = ctx.getKernel("square_distance_float", "square_distance");
		m_cosine_distance_kernel = ctx.getKernel("square_distance_double", "cosine_distance");
		m_cosine_distance_norm_kernel = ctx.getKernel("square_distance_double","cosine_distance_norm");
    if (ctx.memoryType() == Context.HSA_MEMORY)
      WG_COUNT = 4;
    else
      WG_COUNT = 40;
    
		//m_square_distance_kernel_index= ctx.getKernel("square_distance_double", "square_distance_index");
		//m_s_t = ctx.getKernel("square_distance_double", "short_test");
	}

	private void init(Context ctx) {
		StringBuffer data = loadKernel("distance.cl");
		data.append(loadKernel("cosine_dist.cl"));
		ctx.add("square_distance_double", "#define VALUE_TYPE double\n#define COND_TYPE long\n" + data.toString());
		ctx.add("square_distance_float", "#define VALUE_TYPE float\n#define COND_TYPE uint\n" + data.toString());
		
		
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
		
		m_square_distance_kernel.set_global_size(0, WG_SIZE*WG_COUNT);
    m_square_distance_kernel.set_local_size(0, WG_SIZE);
    m_square_distance_kernel.set_arg(0, size);
		m_square_distance_kernel.set_arg(1, test_instance.attributes());
		m_square_distance_kernel.set_arg(2, instance_buffer.attributes());
		m_square_distance_kernel.set_arg(3, min_buffer);
		m_square_distance_kernel.set_arg(4, max_buffer);
		m_square_distance_kernel.set_arg(5, attribute_types);
		m_square_distance_kernel.set_arg(6, indices);
		m_square_distance_kernel.set_arg(7, result);
		m_square_distance_kernel.set_arg(8, dataset.numAttributes());
		m_square_distance_kernel.set_arg(9, 1);
		m_square_distance_kernel.set_arg(10, 0);
		m_square_distance_kernel.invoke();
	}
	
	public void squareDistanceFloat(
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
		m_square_distance_kernel_float.set_global_size(0, WG_SIZE*WG_COUNT);
    m_square_distance_kernel_float.set_local_size(0, WG_SIZE);
    m_square_distance_kernel_float.set_arg(0, size);
    
		m_square_distance_kernel_float.set_arg(1, test_instance.attributes());
		m_square_distance_kernel_float.set_arg(2, instance_buffer.attributes());
		m_square_distance_kernel_float.set_arg(3, min_buffer);
		m_square_distance_kernel_float.set_arg(4, max_buffer);
		m_square_distance_kernel_float.set_arg(5, attribute_types);
		m_square_distance_kernel_float.set_arg(6, indices);
		m_square_distance_kernel_float.set_arg(7, result);
		m_square_distance_kernel_float.set_arg(8, dataset.numAttributes());
		m_square_distance_kernel_float.set_arg(9, 1);
		m_square_distance_kernel_float.set_arg(10, 0);
		m_square_distance_kernel_float.invoke();
	}
	
	public void squareDistanceFloat(
			Instances dataset,
			DenseInstanceBuffer test_instance, 
			DenseInstanceBuffer instance_buffer, 
			Buffer min_buffer, 
			Buffer max_buffer, 
			Buffer attribute_types, 
			Buffer result, 
			int size, 
			Buffer indices, 
			int indices_offset
			) {
		
		
		float[] samples = BufHelper.rbf(instance_buffer.attributes());
		float[] test = BufHelper.rbf(test_instance.attributes());
		float[] min =  BufHelper.rbf(min_buffer);
		float[] max =  BufHelper.rbf(max_buffer);
		
		int global_size = (int)size;
		m_square_distance_kernel_float.set_global_size(0, WG_SIZE*WG_COUNT);
    m_square_distance_kernel_float.set_local_size(0, WG_SIZE);
    m_square_distance_kernel_float.set_arg(0, size);
    
		m_square_distance_kernel_float.set_arg(1, test_instance.attributes());
		m_square_distance_kernel_float.set_arg(2, instance_buffer.attributes());
		m_square_distance_kernel_float.set_arg(3, min_buffer);
		m_square_distance_kernel_float.set_arg(4, max_buffer);
		m_square_distance_kernel_float.set_arg(5, attribute_types);
		m_square_distance_kernel_float.set_arg(6, indices);
		m_square_distance_kernel_float.set_arg(7, result);
		m_square_distance_kernel_float.set_arg(8, dataset.numAttributes());
		m_square_distance_kernel_float.set_arg(9, 1);
		m_square_distance_kernel_float.set_arg(10, indices_offset);
		m_square_distance_kernel_float.invoke();
	}

	public void squareDistanceFloat(Instances dataset,
			DenseInstanceBuffer test_instance, 
			DenseInstanceBuffer instance_buffer, 
			Buffer min_buffer, 
			Buffer max_buffer, 
			Buffer attribute_types, 
			Buffer result 
			) {
		int global_size = (int)instance_buffer.rows();
		m_square_distance_kernel_float.set_global_size(0, WG_SIZE*WG_COUNT);
    m_square_distance_kernel_float.set_local_size(0, WG_SIZE);
    m_square_distance_kernel_float.set_arg(0, global_size);
    
		m_square_distance_kernel_float.set_arg(1, test_instance.attributes());
		m_square_distance_kernel_float.set_arg(2, instance_buffer.attributes());
		m_square_distance_kernel_float.set_arg(3, min_buffer);
		m_square_distance_kernel_float.set_arg(4, max_buffer);
		m_square_distance_kernel_float.set_arg(5, attribute_types);
		m_square_distance_kernel_float.set_arg(6, result);
		m_square_distance_kernel_float.set_arg(7, result);
		m_square_distance_kernel_float.set_arg(8, dataset.numAttributes());
		m_square_distance_kernel_float.set_arg(9, 0);
		m_square_distance_kernel_float.set_arg(10, 0);
		m_square_distance_kernel_float.invoke();
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
		m_square_distance_kernel.set_global_size(0, WG_SIZE*WG_COUNT);
    m_square_distance_kernel.set_local_size(0, WG_SIZE);
    m_square_distance_kernel.set_arg(0, global_size);
    
		m_square_distance_kernel.set_arg(1, test_instance.attributes());
		m_square_distance_kernel.set_arg(2, instance_buffer.attributes());
		m_square_distance_kernel.set_arg(3, min_buffer);
		m_square_distance_kernel.set_arg(4, max_buffer);
		m_square_distance_kernel.set_arg(5, attribute_types);
		m_square_distance_kernel.set_arg(6, result);
		m_square_distance_kernel.set_arg(7, result);
		m_square_distance_kernel.set_arg(8, dataset.numAttributes());
		m_square_distance_kernel.set_arg(9, 0);
		m_square_distance_kernel.set_arg(10, 0);
		m_square_distance_kernel.invoke();
	}
	
	public void cosineDistanceNorm(Instances dataset,
			DenseInstanceBuffer test_instance, 
			DenseInstanceBuffer instance_buffer, 
			Buffer min_buffer, 
			Buffer max_buffer, 
			Buffer attribute_types,  
			Buffer result 
			) {
		int global_size = (int)instance_buffer.rows();
		m_cosine_distance_norm_kernel.set_global_size(0, global_size);
		m_cosine_distance_norm_kernel.set_arg(0, test_instance.attributes());
		m_cosine_distance_norm_kernel.set_arg(1, instance_buffer.attributes());
		m_cosine_distance_norm_kernel.set_arg(2, min_buffer);
		m_cosine_distance_norm_kernel.set_arg(3, max_buffer);
		m_cosine_distance_norm_kernel.set_arg(4, attribute_types);
		m_cosine_distance_norm_kernel.set_arg(5, result);
		m_cosine_distance_norm_kernel.set_arg(6, result);
		m_cosine_distance_norm_kernel.set_arg(7, dataset.numAttributes());
		m_cosine_distance_norm_kernel.set_arg(8, 0);
		
		m_cosine_distance_norm_kernel.invoke();
	}
	
	
	
	public void cosineDistance(Instances dataset,
			DenseInstanceBuffer test_instance, 
			DenseInstanceBuffer instance_buffer, 
			Buffer attribute_types, 
			Buffer result 
			) {
		int global_size = (int)instance_buffer.rows();
		m_cosine_distance_kernel.set_global_size(0, global_size);
		m_cosine_distance_kernel.set_arg(0, test_instance.attributes());
		m_cosine_distance_kernel.set_arg(1, instance_buffer.attributes());
		m_cosine_distance_kernel.set_arg(2, attribute_types);
		m_cosine_distance_kernel.set_arg(3, result);
		m_cosine_distance_kernel.set_arg(4, result);
		m_cosine_distance_kernel.set_arg(5, dataset.numAttributes());
		m_cosine_distance_kernel.set_arg(6, 0);
		m_cosine_distance_kernel.invoke();
	}


}
