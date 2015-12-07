package org.viennacl.binding;

public class Queue {
	public native void flush();
	public native void finish();
	private long m_native_context;
}
