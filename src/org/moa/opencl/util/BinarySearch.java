package org.moa.opencl.util;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;

public class BinarySearch extends AbstractUtil {
	private Context m_context;
	
	public BinarySearch(Context ctx)
	{
		m_context = ctx;
	}
	
	private void init(Context ctx)
	{
	}
	
	public int search(Buffer in, byte[] sample)
	{
		return 0;
	}
	

}
