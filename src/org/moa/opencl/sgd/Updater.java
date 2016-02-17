package org.moa.opencl.sgd;

import org.moa.gpu.SparseMatrix;
import org.viennacl.binding.Buffer;

public interface Updater {

	void simpleUpdate(int rows, Buffer weights, SparseMatrix gradients, double learning_rate);

}