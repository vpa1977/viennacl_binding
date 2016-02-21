package org.moa.opencl.sgd;

import org.moa.gpu.SparseMatrix;
import org.moa.opencl.util.AbstractUtil;
import org.moa.opencl.util.BufHelper;
import org.moa.opencl.util.SparseMatrixOps;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

/** 
 * The updaters can be launched in parallel
 * @author john
 *
 */
public class SimpleUpdater extends AbstractUtil implements Updater {

	private Context m_context;
	private long m_value_size;
	public Buffer m_weights_delta;
	private Buffer m_residual_buffer_small;
	private Buffer m_residual_buffer_large;
	public Buffer m_tau;
	
	private Kernel m_updater;
	private int m_num_attributes;
	private int m_num_classes;
	private Kernel m_read_weights;
	private Kernel m_apply_delta;
	private Buffer m_weights;
	private Kernel m_update_tau;
	private int m_num_batches;
	private double m_learning_rate;

	public SimpleUpdater(
			Context ctx, 
			int num_attributes, 
			int num_classes, 
			int num_batches)
	{
		m_context = ctx;
		m_value_size = DirectMemory.DOUBLE_SIZE;
		double[] initial_tau = new double[num_attributes *num_classes];
		for (int j = 0; j < initial_tau.length; ++j)
			initial_tau[j] = 1;
		m_tau = BufHelper.wb(m_context, initial_tau);
		
		m_residual_buffer_small = new Buffer(ctx, num_attributes *num_classes* num_batches*m_value_size);
		m_residual_buffer_large = new Buffer(ctx, num_attributes *num_classes* num_batches*m_value_size);
		m_weights_delta = new Buffer(ctx, num_attributes*num_classes * DirectMemory.INT_SIZE);
		m_weights = new Buffer(ctx, num_attributes*num_classes * m_value_size);
		m_num_attributes = num_attributes;
		m_num_classes = num_classes;
		m_num_batches = num_batches;
		if (!ctx.hasProgram("sgd_update")) 
		{
			StringBuffer program = new StringBuffer();
			program.append("#define VALUE_TYPE "+ type() + "\n");
			program.append("#define COND_TYPE "+ cond_type() + "\n");
			program.append(loadKernel("updaters.cl"));
			ctx.add("sgd_update", program.toString());
		}
		m_updater = ctx.getKernel("sgd_update", "simple_update");
		m_apply_delta  = ctx.getKernel("sgd_update", "apply_delta");
		m_read_weights = ctx.getKernel("sgd_update", "read_weights");
		m_update_tau = ctx.getKernel("sgd_update", "update_tau");
		m_learning_rate = 0.0001;
	}
	
	private String cond_type() {
		return "long";
	}

	private String type() {
		return "double";
	}

	/** 
	 * SYnchronized to avoid corrupting kernel invokation
	 * @param gradient_buffer
	 * @param batch_number
	 */
	public synchronized  void applyUpdate(Buffer gradient_buffer, 
			int batch_number) {
		m_updater.set_local_size(0, 128);
		m_updater.set_global_size(0, 128*40);
		m_updater.set_local_size(1, 1);
		m_updater.set_global_size(1, m_num_classes);
		
		m_updater.set_arg(0, gradient_buffer);
		m_updater.set_arg(1, m_num_classes);
		m_updater.set_arg(2, m_num_attributes);
		m_updater.set_arg(3, m_tau);
		m_updater.set_arg(4, m_residual_buffer_small);
		m_updater.set_arg(5, m_residual_buffer_large);
		m_updater.set_arg(6, m_weights_delta);
		m_updater.set_arg(7, batch_number);
		m_updater.invoke();
		
	}

	public void setTau(Buffer tau) {
		tau.copyTo(m_tau);
	}

	public Buffer getErrorSmall() {
		return m_residual_buffer_small;
	}
	
	public Buffer getErrorLarge() 
	{
		return m_residual_buffer_large;
	}
	
	public Buffer getWeightsDelta()
	{
		return m_weights_delta;
	}

	public void applyWeightsDelta() {
		m_apply_delta.set_global_size(0, 40 * 128);
		m_apply_delta.set_local_size(0, 128);
		m_apply_delta.set_arg(0,  m_weights);
		m_apply_delta.set_arg(1, m_num_attributes);
		m_apply_delta.set_arg(2, m_num_classes);
		m_apply_delta.set_arg(3, m_weights_delta);
		m_apply_delta.set_arg(4, m_tau);
		m_apply_delta.set_arg(5, m_learning_rate);
		m_apply_delta.invoke();
	}

	/** 
	 * SYnchronized to avoid corrupting kernel parameters
	 * @param gradient_buffer
	 * @param batch_number
	 */
	public synchronized void readWeights(Buffer weights)
	{
		//double[] w = BufHelper.rb(m_weights);
		//double[] tau = BufHelper.rb(m_tau);
		//int[] delta = BufHelper.rbi(m_weights_delta);
		m_read_weights.set_global_size(0, 40 * 128);
		m_read_weights.set_local_size(0, 128);
		m_read_weights.set_arg(0,weights);
		m_read_weights.set_arg(1,  m_weights);
		m_read_weights.set_arg(2, (int)m_num_attributes);
		m_read_weights.set_arg(3, (int)m_num_classes);
		m_read_weights.set_arg(4, m_weights_delta);
		m_read_weights.set_arg(5, m_tau);
		m_read_weights.set_arg(6,  (double)m_learning_rate);
		m_read_weights.invoke();
		//double[] res = BufHelper.rb(weights);
		//System.out.println();
	}

	public Buffer getWeights() {
		return m_weights;
	}

	public void updateTau() {
		m_update_tau.set_global_size(0, 40 * 128);
		m_update_tau.set_local_size(0, 128);
		m_update_tau.set_arg(0,  m_tau);
		m_update_tau.set_arg(1,  m_residual_buffer_small);
		m_update_tau.set_arg(2,  m_residual_buffer_large);
		m_update_tau.set_arg(3, m_num_batches);
		m_update_tau.set_arg(4, m_num_attributes);
		m_update_tau.set_arg(5, m_num_classes);
		m_update_tau.invoke();
		
	}

	public Buffer getTau() {
		return m_tau;
	}
}
