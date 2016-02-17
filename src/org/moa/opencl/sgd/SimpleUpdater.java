package org.moa.opencl.sgd;

import org.moa.gpu.SparseMatrix;
import org.moa.opencl.util.AbstractUtil;
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
	private Buffer m_residual_buffer;
	private Buffer m_gradient_buffer;
	private SparseMatrixOps m_ops;
	private Kernel m_updater;
	private int m_columns;

	public SimpleUpdater(Context ctx, int num_attributes)
	{
		m_context = ctx;
		m_value_size = DirectMemory.DOUBLE_SIZE;
		m_gradient_buffer = new Buffer(ctx, num_attributes * m_value_size);
		m_ops = new SparseMatrixOps(ctx);
		m_columns = num_attributes;
		if (!ctx.hasProgram("sgd_update")) 
		{
			ctx.add("sgd_update", loadKernel("updaters.cl").toString());
		}
		m_updater = ctx.getKernel("sgd_Update", "simple_update");
	}
	
	/* (non-Javadoc)
	 * @see org.moa.opencl.sgd.Updater#update(int, org.viennacl.binding.Buffer, org.moa.gpu.SparseMatrix, double)
	 */
	@Override
	public void simpleUpdate(int rows, Buffer weights, SparseMatrix gradients, double learning_rate)
	{
		m_ops.columnSum(gradients, m_gradient_buffer);
		applyUpdate(weights,m_gradient_buffer, learning_rate, rows);
		
		
	}

	private void applyUpdate(Buffer weights, Buffer gradient_buffer, double learning_rate, int rows) {
		m_updater.set_local_size(0, 128);
		m_updater.set_global_size(0, 128*40);
		m_updater.set_arg(0, m_columns);
		m_updater.set_arg(1, rows);
		m_updater.set_arg(2, learning_rate);
		m_updater.set_arg(3, gradient_buffer);
		m_updater.set_arg(4, weights);
		
	}
}
