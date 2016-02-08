package org.moa.opencl.util;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

public class NarySearch extends AbstractUtil {

	public final int NUM_SEARCH_GROUPS = 2*2560/64;//384/64;
	private Context m_context;
	private Kernel m_search_kernel;
	private Buffer m_search_result;
	private Buffer m_search_bounds;
	/** 
	 * 
	 * @param ctx - context
	 * @param checkBounds - check if search value below or above bounds in the first search phase
	 */
	public NarySearch(Context ctx, boolean checkBounds)
	{
		m_context = ctx;
		//m_search_result = new Buffer(m_context, 3 * DirectMemory.INT_SIZE);
		m_search_bounds = new Buffer(m_context, 3 * DirectMemory.INT_SIZE);
		if (ctx.memoryType() != ctx.MAIN_MEMORY)
		{
			if (!m_context.hasProgram("nary_search"))
				init(m_context);
			m_search_kernel = m_context.getKernel("nary_search", "search");
		}
	}
	
	private void init(Context m_context2) {
		StringBuffer kernel = loadKernel("search.cl");
		m_context.add("nary_search", kernel.toString());
	}

	public int getSearchPos() 
	{
		m_search_bounds.mapBuffer(Buffer.READ);
		int pos = m_search_bounds.readInt(0);
		m_search_bounds.commitBuffer();
		return pos;
	}
	/** 
	 * Perform a n-ry search
	 * @param keyValues - char array of key values
	 * @param keyOrder - int array with keys sort order
	 * @param searchTerm - char array with a search key
	 * @param keyLength - size of the key
	 * @return index into keyOrder buffer 
	 */
	public void search(Buffer keyValues, Buffer keyOrder, Buffer searchTerm, int keyLength, int sequenceLength, boolean natural)
	{
		if (m_context.memoryType() == Context.MAIN_MEMORY )
			throw new RuntimeException("Use Arrays.binarySearch()");
		m_search_kernel.set_local_size(0, 64); // 1 wavefront
		m_search_kernel.set_global_size(0,64  * NUM_SEARCH_GROUPS);
		
		m_search_bounds.mapBuffer(Buffer.WRITE);
		m_search_bounds.writeArray(0, new int[]{0, sequenceLength-1, 0});
		
		m_search_bounds.commitBuffer();
		
		m_search_kernel.set_arg(0, keyValues);
		m_search_kernel.set_arg(1, keyOrder);
		m_search_kernel.set_arg(2, searchTerm);
		m_search_kernel.set_arg(3, keyLength);
		m_search_kernel.set_arg(4, m_search_bounds);
		m_search_kernel.set_arg(5,  natural ? 1 : 0 );
		
		int count =(int) (Math.log(sequenceLength)/Math.log(NUM_SEARCH_GROUPS)) + 1;
		for (int i = 0; i < count; ++i)
		{
			m_search_kernel.invoke();
			/*int[] test = new int[3];
			m_search_bounds.mapBuffer(Buffer.READ);
			m_search_bounds.readArray(0,test);
			m_search_bounds.commitBuffer();*/
		}
			
		
		
	}
	
}
