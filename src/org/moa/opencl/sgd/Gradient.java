package org.moa.opencl.sgd;

import org.moa.gpu.SparseMatrix;
import org.moa.opencl.util.AbstractUtil;
import org.viennacl.binding.Buffer;

public abstract class Gradient extends AbstractUtil {
	public abstract Buffer computeGradient(Buffer classValues, SparseMatrix minibatch, Buffer weights);
}
