package org.viennacl.binding;

public class Kernel {
	public native void set_local_size(int i, int j);
	public native void set_global_size(int dim, int size);
	 
	public native void set_arg(int arg, Buffer param);
	public native void set_arg(int arg, int param);
	public native void set_arg(int arg, long param);
	public native void invoke(Object... params);
	public native void invoke();
	
	public void finalize()
	{
		m_native_context = 0;
	}
	public void setContext(Context ctx) {
		m_context = ctx;
	}
	
	private long m_native_context;
	private Context m_context;
	
	
}
