package org.moa.opencl.sgd;

import java.util.concurrent.Semaphore;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.MappedFile;

public class SharedSemaphores {
  
  static {
    System.loadLibrary("viennacl-java-binding");
  }
	private MappedFile m_semaphores;
	private int m_size;

	public SharedSemaphores(String id, int num_semaphores) throws Exception
	{
		m_size = (int)(num_semaphores*DirectMemory.LONG_SIZE);
		m_semaphores = new MappedFile(id, m_size);
		
	}
	
	public boolean getValue(int offset)
	{
		if (offset >= m_size || offset < 0)
			throw new RuntimeException("range check");
		return nativeGet(m_semaphores.getAddr(), offset) != 0;
	}
  
  private native void nativeSet(long address, int offset, int val);
  private native int nativeGet(long address, int offset);
	
	public void setValue(int offset, boolean b)
	{
		if (offset >= m_size || offset < 0)
			throw new RuntimeException("range check");
  //  System.out.println("Setting semaphore "+ offset +" value "+ b + " at address " + (m_semaphores.getAddr() + DirectMemory.LONG_SIZE * offset));
    nativeSet(m_semaphores.getAddr(), offset, b ? 33 : 0 );
	}
  
  public static void main(String[] args) throws Exception
  {
      SharedSemaphores test = new SharedSemaphores("test", 2);
      test.setValue(0, true);
      System.out.println(test.getValue(0));
      test.setValue(1, true);
      System.out.println(test.getValue(0));
      test.setValue(1, false);
      System.out.println(test.getValue(0));
  }

}
