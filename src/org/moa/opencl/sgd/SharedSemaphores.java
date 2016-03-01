package org.moa.opencl.sgd;

import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.MappedFile;

public class SharedSemaphores {
	private MappedFile m_semaphores;
	private int m_size;

	public SharedSemaphores(String id, int num_semaphores) throws Exception
	{
		m_semaphores = new MappedFile(id, num_semaphores);
		m_size = num_semaphores;
	}
	
	public boolean getValue(int offset)
	{
		if (offset >= m_size || offset < 0)
			throw new RuntimeException("range check");
		return DirectMemory.readByte(m_semaphores.getAddr() + offset) != 0;
	}
	
	public void setValue(int offset, boolean b)
	{
		if (offset >= m_size || offset < 0)
			throw new RuntimeException("range check");
		DirectMemory.writeByte(m_semaphores.getAddr() + offset,  b ? (byte)1 :(byte)0);
	}

}
