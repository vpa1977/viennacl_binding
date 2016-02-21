package org.moa.opencl.sgd;

import org.moa.gpu.SparseMatrix;
import org.viennacl.binding.Buffer;

public interface Updater {

	public 	void applyUpdate(Buffer gradient_buffer, int batch_number);

}