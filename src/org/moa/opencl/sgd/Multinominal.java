package org.moa.opencl.sgd;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.SparseInstanceBuffer;
import org.moa.opencl.util.AbstractUtil;
import org.moa.opencl.util.BufHelper;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

import weka.core.Instances;

public class Multinominal extends AbstractUtil {
	
	private int m_num_classes;
	private int m_num_attributes;
	private int m_minibatch_size;
	private Buffer m_dot_products;
	private Kernel m_compute_margin_bounds;
	private Buffer m_minibatch_gradients;
	private Kernel m_multinominal_hinge;
	private Kernel m_apply_multiplier;
	private Kernel m_reduce_to_minibatch_dense;
	private Buffer m_minibatch_gradient_temp;
	private Context m_context;


	public Multinominal(Context ctx, int num_classes, int num_attributes,  int minibatch_size) 
	{
		if (!ctx.hasProgram("multinominal"))
			init(ctx);
		m_num_classes = num_classes;
		m_num_attributes = num_attributes;
		m_minibatch_size = minibatch_size;
		//m_compute_margin_bounds = ctx.getKernel("multinominal", "compute_margin_bounds");
		m_multinominal_hinge = ctx.getKernel("multinominal", "multinominal_hinge");
		m_apply_multiplier =ctx.getKernel("multinominal", "multiply_and_makedense");
		m_reduce_to_minibatch_dense = ctx.getKernel("multinominal", "reduce_to_minibatch_dense");
		m_dot_products = new Buffer(ctx, m_minibatch_size * num_classes * typeSize());
		m_minibatch_gradients = new Buffer(ctx, typeSize() * num_attributes * num_classes); // 1 gradient per each class
		m_minibatch_gradient_temp = new Buffer(ctx, typeSize() * num_attributes); // 1 gradient per each class
		m_context = ctx;
		m_reduce_buffer = new Buffer(m_context, num_classes*num_attributes * typeSize());
	}
	
	public void computeGradient(Instances dataset, SparseInstanceBuffer instance_buffer, Buffer weights, Buffer bias)
	{

		computeDotProducts(
		 	instance_buffer.getColumnIndices(),
			 instance_buffer.getRowJumper(),
			 instance_buffer.getElements(), 
			 instance_buffer.getRowBlocks(),
			 instance_buffer.getRowBlockNum(), 
			 instance_buffer.getColumnCount(),
			 m_minibatch_size, 
			 instance_buffer.getRowPostion(), 
			  m_num_classes, weights,
			  m_dot_products);
		computeMultinominalHinge(m_dot_products, instance_buffer.classes(), bias, dataset.numClasses(), m_minibatch_size);
		computeReduceToMinibatch(m_minibatch_gradients, m_dot_products, 1, dataset.numClasses(), 
				m_minibatch_size, dataset.numAttributes(), instance_buffer);
	}
	
	public double[] predict(Instances dataset, DenseInstanceBuffer instance_buffer, Buffer weights, Buffer bias)
	{
		computeDotProducts(
				instance_buffer.attributes(),
				dataset.numAttributes(),
				1, 
				  m_num_classes, weights,
				  m_dot_products);
		double[] result = new double[m_num_classes];
		double[] dot_products = BufHelper.rb(m_dot_products);
		for (int i = 0; i < result.length ; ++i)
			result[i] = dot_products[i] <= 0 ? 0 :1;
		return result;
	}
	
	public void computeGradient(Instances dataset, DenseInstanceBuffer instance_buffer, Buffer weights, Buffer bias)
	{
		computeDotProducts(
			instance_buffer.attributes(),
			dataset.numAttributes(),
			m_minibatch_size, 
			  m_num_classes, weights,
			  m_dot_products);
		
//		double[] attrs = BufHelper.rb(instance_buffer.attributes());
//		double[] w = BufHelper.rb(weights);
//		double[] dot_products = BufHelper.rb(m_dot_products);
//		double[] classes = BufHelper.rb(instance_buffer.classes());
		computeMultinominalHinge(m_dot_products, instance_buffer.classes(), bias, dataset.numClasses(), m_minibatch_size);
	//	dot_products = BufHelper.rb(m_dot_products);
		computeReduceToMinibatch(m_minibatch_gradients, m_dot_products, 1, dataset.numClasses(), 
				m_minibatch_size, dataset.numAttributes(), instance_buffer);
		
	}
	
	
	public Buffer getComputedGradients() 
	{
		return m_minibatch_gradients;
	}
	
	public synchronized void computeMultinominalHinge(Buffer dotProduct, Buffer classes, Buffer bias, int num_classes, int num_rows)
	{
		m_multinominal_hinge.set_local_size(1, 1);
		m_multinominal_hinge.set_global_size(1, num_classes);
		m_multinominal_hinge.set_local_size(0,  128);
		m_multinominal_hinge.set_global_size(0,  128*40);
		m_multinominal_hinge.set_arg(0,  dotProduct);
		m_multinominal_hinge.set_arg(1,  classes);
		m_multinominal_hinge.set_arg(2,  bias);
		m_multinominal_hinge.set_arg(3, num_classes);
		m_multinominal_hinge.set_arg(4, num_rows);
		m_multinominal_hinge.invoke();
	}
	
	
	private Buffer m_reduce_buffer;
	
	public synchronized void computeReduceToMinibatch(Buffer minbatchGradients, 
			Buffer dotProduct, 
			int num_batches, 
			int num_classes, 
			int batch_size, 
			int num_attributes, 
			SparseInstanceBuffer source)
	{
		if (true)
			throw new RuntimeException("Not tested");
		minbatchGradients.fill((byte)0);
		for (int class_index = 0; class_index < num_classes ; ++class_index)
		{
			
			m_reduce_buffer.fill((byte)0);
			m_apply_multiplier.set_local_size(0, 256);
			m_apply_multiplier.set_global_size(0, 256* source.getRowBlockNum());
			m_apply_multiplier.set_arg(0, m_reduce_buffer);
			
			m_apply_multiplier.set_arg(1, source.getRowJumper());
			m_apply_multiplier.set_arg(2, source.getColumnIndices());
			m_apply_multiplier.set_arg(3, source.getRowBlocks());
			m_apply_multiplier.set_arg(4, source.getRowBlockNum());
			m_apply_multiplier.set_arg(5, source.getElements());
			
			m_apply_multiplier.set_arg(6, dotProduct);
			m_apply_multiplier.set_arg(7, class_index);
			m_apply_multiplier.set_arg(8, num_attributes );
			m_apply_multiplier.invoke();
			
			computeReduction(m_reduce_buffer, 
					source.getColumnCount(),
					m_num_attributes, 
					m_minibatch_gradient_temp);
			m_minibatch_gradient_temp.copyTo(minbatchGradients, class_index * num_attributes * typeSize());
		}
		
		
		
	}
	
	public synchronized void computeReduceToMinibatch(Buffer minbatchGradients, Buffer dotProduct, int num_batches, int num_classes, int batch_size, int num_attributes, 
			DenseInstanceBuffer source)
	{
		minbatchGradients.fill((byte)0);
		m_reduce_to_minibatch_dense.set_local_size(0, 256);
		m_reduce_to_minibatch_dense.set_global_size(0, 256*(1 + (int)(num_attributes/256)));
		m_reduce_to_minibatch_dense.set_local_size(1,  1);
		m_reduce_to_minibatch_dense.set_global_size(1,  num_classes);

		m_reduce_to_minibatch_dense.set_arg(0,  minbatchGradients);
		m_reduce_to_minibatch_dense.set_arg(1,  dotProduct);
		m_reduce_to_minibatch_dense.set_arg(2,  (int)m_num_classes);
		m_reduce_to_minibatch_dense.set_arg(3,  (int)batch_size);
		m_reduce_to_minibatch_dense.set_arg(4,  (int)num_attributes);
		m_reduce_to_minibatch_dense.set_arg(5,  source.attributes());
		m_reduce_to_minibatch_dense.invoke();
	}

	 
	/**     
	 * computes dot product matrix for sparse matrix
	 * @param column_indices
	 * @param row_jumper
	 * @param elements
	 * @param row_blocks
	 * @param row_block_count
	 * @param columns
	 * @param rows
	 * @param element_count
	 * @param num_classes
	 * @param weights
	 * @param margins
	 */
	public synchronized  native void computeDotProducts(Buffer column_indices, 
										   Buffer row_jumper, 
										   Buffer elements,
										   Buffer row_blocks, 
										   int row_block_count,
										   int columns, 
										   int rows, 
										   int element_count,
										   int num_classes,
										   Buffer weights, Buffer margins);

	/**     
	 * computes column reduction for sparse matrix
	 * @param column_indices
	 * @param row_jumper
	 * @param elements
	 * @param row_blocks
	 * @param row_block_count
	 * @param columns
	 * @param rows
	 * @param element_count
	 * @param num_classes
	 * @param weights
	 * @param margins
	 */
	public synchronized  native void computeReduction( 
										   Buffer elements,
										   int columns, 
										   int rows, 
										   Buffer reduction_result);
	
	/** 
	 * Computes dot product matrix for dense buffer
	 * @param attributes
	 * @param columns
	 * @param rows
	 * @param element_count
	 * @param num_classes
	 * @param weights
	 * @param margins
	 */
	public synchronized native void computeDotProducts(Buffer attributes, 
			   int columns, 
			   int rows, 
			   int num_classes,
			   Buffer weights, Buffer margins);

/*	public void computeMarginBounds(Buffer margins, Buffer bounds)
	{
		m_compute_margin_bounds.set_local_size(0, 256);
		m_compute_margin_bounds.set_global_size(0, 256*40);
		m_compute_margin_bounds.set_arg(0,  margins);
		m_compute_margin_bounds.set_arg(1,  m_num_classes);
		m_compute_margin_bounds.set_arg(2,  m_num_rows);
		m_compute_margin_bounds.set_arg(3,  bounds);
		m_compute_margin_bounds.invoke();
		
	}*/
	public String type() 
	{
		return "double";
	}
	
	public String cond_type() 
	{
		return "ulong";
	}
	
	public int typeSize() 
	{
		return (int) DirectMemory.DOUBLE_SIZE;
	}
	
	private void init(Context ctx) {
		StringBuffer program = new StringBuffer();
		program.append("#define VALUE_TYPE " + type() + "\n");
		program.append("#define COND_TYPE " + cond_type() + "\n");
		program.append(loadKernel("multinominal.cl"));
		ctx.add("multinominal", program.toString());
	}
	
}
