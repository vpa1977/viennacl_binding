package moa.classifiers.gpu.zorder;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;

public class ZOrderSequence {
	
	private Buffer m_indices;
	private Buffer m_code;
	private Context m_context;
	
	public ZOrderSequence(Context ctx, Buffer indices, Buffer code)
	{
		m_context = ctx;
		m_indices = new Buffer(ctx, indices.byteSize());
		m_code = new Buffer(ctx, code.byteSize());
		indices.copyTo(m_indices);
		code.copyTo(m_code);
	}
	
	Buffer code() 
	{
		return m_code;
	}
	
	Buffer indices() 
	{
		return m_indices;
	}

}
