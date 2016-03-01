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
	
	private Context m_context;
	private Kernel m_reduce_to_minibatch_sparse;


	public Multinominal(Context ctx, int num_classes, int num_attributes,  int minibatch_size) 
	{
		if (!ctx.hasProgram("multinominal"))
			init(ctx);
		m_num_classes = num_classes;
		m_num_attributes = num_attributes;
		m_minibatch_size = minibatch_size;
		//m_compute_margin_bounds = ctx.getKernel("multinominal", "compute_margin_bounds");
		m_multinominal_hinge = ctx.getKernel("multinominal", "multinominal_hinge");
		m_reduce_to_minibatch_dense = ctx.getKernel("multinominal", "reduce_to_minibatch_dense");
		m_reduce_to_minibatch_sparse = ctx.getKernel("multinominal","reduce_to_minibatch_sparse");
		
		m_dot_products = new Buffer(ctx, m_minibatch_size * num_classes * typeSize());
		m_minibatch_gradients = new Buffer(ctx, typeSize() * num_attributes * num_classes); // 1 gradient per each class
	
		m_context = ctx;
	}
	
	public void computeGradient(Instances dataset, SparseInstanceBuffer instance_buffer, Buffer weights)
	{
		//BufHelper.print("Elements " , instance_buffer.getElements(), instance_buffer.getRowPostion());
		//BufHelper.print("Weights " , weights,  m_num_attributes* m_num_classes);
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
		//BufHelper.print("Dot Products: ", m_dot_products, dataset.numClasses());
		//double[] classes = BufHelper.rb(instance_buffer.classes());
		computeMultinominalHinge(dataset.classIndex(), weights, 
				m_dot_products, instance_buffer.classes(), 
				dataset.numClasses(), m_minibatch_size, dataset.numAttributes());
		//BufHelper.print("Hinge: ", m_dot_products, dataset.numClasses());
		computeReduceToMinibatch(dataset.classIndex(),m_minibatch_gradients, m_dot_products,  dataset.numClasses(), 
				 dataset.numAttributes(), instance_buffer);

		//double[] gradients = BufHelper.rb(m_minibatch_gradients);
		//BufHelper.print("minibatch_gradients", m_minibatch_gradients, gradients.length);

	}
	
	public double[] predict(Instances dataset, DenseInstanceBuffer instance_buffer, Buffer weights)
	{

		// class value is replaced by 1 to use W0
		computeDotProducts(
				instance_buffer.attributes(),
				dataset.numAttributes(),
				1, 
				  m_num_classes, weights,
				  m_dot_products);
		double[] result = new double[m_num_classes];
		double[] dot_products = BufHelper.rb(m_dot_products);
		for (int i = 0; i < result.length ; ++i)
			result[i] = dot_products[i]   <= 0 ? 0 :1;
		return result;
	}
	
	public void computeGradient(Instances dataset, DenseInstanceBuffer instance_buffer, Buffer weights)
	{
	//	BufHelper.print("dense Elements " , instance_buffer.attributes(), m_num_attributes);
	//	BufHelper.print("dense Weights " , weights, m_num_attributes* m_num_classes);

		computeDotProducts(
			instance_buffer.attributes(),
			dataset.numAttributes(),
			m_minibatch_size, 
			  m_num_classes, weights,
			  m_dot_products);
		
	//	BufHelper.print("dense dot_products", m_dot_products, (int)(m_dot_products.byteSize()/DirectMemory.DOUBLE_SIZE));
		if (false)
		{
			computeHingeCPU(dataset, instance_buffer, weights);
		}
		else
		{
		computeMultinominalHinge(dataset.classIndex(), weights, 
				m_dot_products, instance_buffer.classes(), dataset.numClasses(), m_minibatch_size, 
				dataset.numAttributes());
		}
	//	BufHelper.print("dense hinge", m_dot_products, (int)(m_dot_products.byteSize()/DirectMemory.DOUBLE_SIZE));
		
		if (false)
		{
			reduceToMinibatchCPU(dataset.classIndex(),instance_buffer);
			return;
		}
		
		computeReduceToMinibatch(dataset.classIndex(), m_minibatch_gradients, m_dot_products, dataset.numClasses(), 
				 dataset.numAttributes(), instance_buffer);
		//double[] gradients = BufHelper.rb(m_minibatch_gradients);
		//BufHelper.print("dense minibatch_gradients", m_minibatch_gradients, gradients.length);
		
	}

	private void computeHingeCPU(Instances dataset, DenseInstanceBuffer instance_buffer, Buffer weights) {
		double[] class_values = BufHelper.rb(instance_buffer.classes());
		double[] dp = BufHelper.rb(m_dot_products);
		double[] weights_cpu = BufHelper.rb(weights);
		for (int row = 0; row < m_minibatch_size; ++row)
		for (int c = 0; c < m_num_classes; ++c)
		{
			double mult = (int)class_values[row] == c ? 1 : -1;
			double z = mult * (dp[row * m_num_classes + c ] + weights_cpu[ c * m_num_attributes + dataset.classIndex()]);
			dp[row * m_num_classes + c ] = z < 1 ? 1 : 0;
			dp[row * m_num_classes + c ] *=mult; 
		}
		m_dot_products.mapBuffer(Buffer.WRITE);
		m_dot_products.writeArray(0,  dp);
		m_dot_products.commitBuffer();
	}

	private void reduceToMinibatchCPU(int class_idx, DenseInstanceBuffer instance_buffer) {
		double[] attributes = BufHelper.rb(instance_buffer.attributes());
		double[] gradient = new double[m_num_classes * m_num_attributes];
		double [] dp = BufHelper.rb(m_dot_products);
		for (int c = 0; c < m_num_classes; ++c)
		for (int i = 0; i < m_minibatch_size ; ++i)
		{
			for (int j = 0; j < m_num_attributes; ++j)
			{
				int att_id = j + m_num_attributes * i;
				double upd = (j == class_idx ? 1 :attributes[att_id]) * dp[c + i*m_num_classes];
				int idx = c*m_num_attributes + j;
				gradient[idx]+= upd;
			}
		}
		m_minibatch_gradients.mapBuffer(Buffer.WRITE);
		m_minibatch_gradients.writeArray(0, gradient);
		m_minibatch_gradients.commitBuffer();
		
	}
	
	
	public Buffer getComputedGradients() 
	{
		return m_minibatch_gradients;
	}
	
	public synchronized void computeMultinominalHinge(int classIndex,
			Buffer weights,
			Buffer dotProduct, Buffer classes, int num_classes, int num_rows, int num_attributes)
	{
		m_multinominal_hinge.set_local_size(1, 1);
		m_multinominal_hinge.set_global_size(1, num_classes);
		m_multinominal_hinge.set_local_size(0,  128);
		m_multinominal_hinge.set_global_size(0,  128*40);
		m_multinominal_hinge.set_arg(0,  dotProduct);
		m_multinominal_hinge.set_arg(1,  classes);
		m_multinominal_hinge.set_arg(2,  weights);
		m_multinominal_hinge.set_arg(3,  classIndex);
		m_multinominal_hinge.set_arg(4, num_classes);
		m_multinominal_hinge.set_arg(5, num_rows);
		m_multinominal_hinge.set_arg(6, num_attributes);
		m_multinominal_hinge.invoke();
	}
	
	
	public synchronized void computeReduceToMinibatch(int classIndex,Buffer gradients, 
			Buffer dotProduct, 
			int num_classes, 
			int num_attributes, 
			SparseInstanceBuffer source)
	{
		gradients.fill((byte)0);
		m_reduce_to_minibatch_sparse.set_global_size(0, 256);
		m_reduce_to_minibatch_sparse.set_local_size(0, 256);
		m_reduce_to_minibatch_sparse.set_local_size(1,  1);
		m_reduce_to_minibatch_sparse.set_global_size(1,  num_classes);
		//double[] ss = BufHelper.rb(source.getElements());
		m_reduce_to_minibatch_sparse.set_arg(0,gradients );
		m_reduce_to_minibatch_sparse.set_arg(1,  dotProduct);
		m_reduce_to_minibatch_sparse.set_arg(2,  num_classes);
		m_reduce_to_minibatch_sparse.set_arg(3,  m_num_attributes);
		m_reduce_to_minibatch_sparse.set_arg(4,  source.getRowJumper());
		m_reduce_to_minibatch_sparse.set_arg(5,  source.getColumnIndices());
		m_reduce_to_minibatch_sparse.set_arg(6,  source.getRowBlocks());
		m_reduce_to_minibatch_sparse.set_arg(7,  source.getRowBlockNum());
		m_reduce_to_minibatch_sparse.set_arg(8,  source.getElements());
		m_reduce_to_minibatch_sparse.set_arg(10, m_minibatch_size);
		m_reduce_to_minibatch_sparse.set_arg(11, classIndex);
		for (int row = 0; row < m_minibatch_size; ++ row)
		{
			m_reduce_to_minibatch_sparse.set_arg(9, row);
			m_reduce_to_minibatch_sparse.invoke();
			//double[] temp = BufHelper.rb(gradients);
		//	System.out.println();
		}
	}
	
	


	
	
	public synchronized void computeReduceToMinibatch(int classIndex, Buffer minbatchGradients, Buffer dotProduct, int num_classes, int num_attributes, 
			DenseInstanceBuffer source)
	{
		minbatchGradients.fill((byte)0);
		m_reduce_to_minibatch_dense.set_local_size(0, 256);
		m_reduce_to_minibatch_dense.set_global_size(0, 256*40);
		m_reduce_to_minibatch_dense.set_local_size(1,  1);
		m_reduce_to_minibatch_dense.set_global_size(1,  num_classes);

		m_reduce_to_minibatch_dense.set_arg(0,  minbatchGradients);
		m_reduce_to_minibatch_dense.set_arg(1,  dotProduct);
		m_reduce_to_minibatch_dense.set_arg(2,  (int)m_num_classes);
		m_reduce_to_minibatch_dense.set_arg(3,  (int)m_minibatch_size);
		m_reduce_to_minibatch_dense.set_arg(4,  (int)num_attributes);
		m_reduce_to_minibatch_dense.set_arg(5,  source.attributes());
		m_reduce_to_minibatch_dense.set_arg(6,  classIndex);
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
