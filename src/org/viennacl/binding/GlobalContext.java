package org.viennacl.binding;

public class GlobalContext {
	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
	public static Context context() {
		return static_context;
	}
	
	private static Context static_context = new Context(Context.Memory.OPENCL_MEMORY, null);

}
