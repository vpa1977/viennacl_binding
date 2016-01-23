package org.viennacl.binding;

import java.util.HashSet;

public class Context {
	public static final  int MAIN_MEMORY = 0;
	public static final  int OPENCL_MEMORY = 1;
	public static final  int HSA_MEMORY = 2;
	public enum Memory {
		MAIN_MEMORY,
		OPENCL_MEMORY,
		HSA_MEMORY
	}
  
  public static final Memory DEFAULT_MEMORY = Memory.HSA_MEMORY;
  
  
	public Context(Context.Memory mem_type, String params)
	{
		m_mem_type = mem_type;
		switch(mem_type)
		{
		case MAIN_MEMORY:init(MAIN_MEMORY, params);
			break;
		case OPENCL_MEMORY:init(OPENCL_MEMORY, params);
			break;	
		case HSA_MEMORY: init(HSA_MEMORY, params);
			break;
		}
	}
	public void finalize() 
	{
		freePrograms();
		release();
	}
	public void freePrograms() {
		for (String p : m_programs)
			removeProgram(p);
		m_programs.clear();
	}
	
	public void add(String program, String text)
	{
		addProgram(program, text);
		m_programs.add(program);
	}
	
	public int memoryType() {
		switch(m_mem_type)
		{
		case MAIN_MEMORY:return MAIN_MEMORY;
		case OPENCL_MEMORY:return 1;
		case HSA_MEMORY: return 2;
		}
		return -1;
	}
	public boolean hasProgram(String string) {
		return m_programs.contains(string);
	}
	
	public  Kernel getKernel(String program, String name)
	{
		Kernel kernelObject = new Kernel();
		return nativeGetKernel(program, name, kernelObject);
	}
	
	public native void finishDefaultQueue();
	

	
	public native void release();
	private native void init(int mem_type, String params);
	
	private native void addProgram(String program, String text);
	
	private native Kernel nativeGetKernel(String program, String name, Kernel kernelObject);
	public native void removeProgram(String string);
	public native Queue createQueue();
	
	private long m_native_context;
	Context.Memory m_mem_type;
	private HashSet<String> m_programs = new HashSet<String>();

	
}
