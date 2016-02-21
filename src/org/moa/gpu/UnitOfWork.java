package org.moa.gpu;

import weka.core.Instance;

public interface UnitOfWork {
	
	public boolean append(Instance inst);
	public void reset();
	public void begin(int mode);
	public void commit();
}
