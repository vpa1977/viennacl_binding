package org.moa.opencl.sgd;

import org.moa.gpu.SparseMatrix;
import org.moa.opencl.util.SparseMatrixOps;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

public class HingeGradient extends Gradient {

	private Context m_context;
	private SparseMatrixOps m_matrix_mult;
	private long m_value_size;
	private Buffer m_factors_buffer;
	private Buffer m_loss_buffer;
	private Kernel m_hinge_kernel;

	public HingeGradient(Context ctx, int num_attributes, int rows)
	{
		m_value_size = DirectMemory.DOUBLE_SIZE;
		m_context = ctx;
		m_matrix_mult = new SparseMatrixOps(ctx);
		m_factors_buffer = new Buffer(ctx, num_attributes * m_value_size);
		m_loss_buffer = new Buffer(ctx, num_attributes * m_value_size);
		if(!ctx.hasProgram("hinge_gradient"))
		{
			ctx.add("hinge_gradient", type() + 
									   loadKernel("hinge.cl").toString());
		}
		m_hinge_kernel = ctx.getKernel("hinge_gradient", "hinge");
	}


	private String type() {
		return "#define VALUE_TYPE double\n" +
				"#define COND_TYPE long\n";
	}
	
	
	/** 
	 * 
	 * @param classValues - class values for mini batch {0|1}
	 * @param minibatch - sparse matrix with attribute values
	 * @param weights - weights
	 * @param factors - temporary buffer for factors calcuation
	 * @return member buffer containing loss for the current mini-batch
	 */
	public Buffer computeGradient(Buffer classValues, SparseMatrix minibatch, Buffer weights) {
		
		m_matrix_mult.mult(minibatch, weights, m_factors_buffer);
		m_hinge_kernel.set_local_size(0, 256);
		m_hinge_kernel.set_global_size(0, 256 *  minibatch.getRowBlockNum());
		m_hinge_kernel.set_arg(0, classValues);
		m_hinge_kernel.set_arg(1, m_factors_buffer);
		m_hinge_kernel.set_arg(2, minibatch.getRowJumper());
		m_hinge_kernel.set_arg(3, minibatch.getColumnIndices());
		m_hinge_kernel.set_arg(4, minibatch.getRowBlocks());
		m_hinge_kernel.set_arg(5, minibatch.getRowBlockNum());
		m_hinge_kernel.set_arg(6, minibatch.getElements());
		m_hinge_kernel.set_arg(7, m_loss_buffer);
		m_hinge_kernel.invoke();
		return m_loss_buffer;
	}

}
