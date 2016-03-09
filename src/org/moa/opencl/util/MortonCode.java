package org.moa.opencl.util;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

public class MortonCode extends AbstractUtil {
	
	private static final int BYTE_COUNT = 256;
	
	private Buffer m_lookup_table;

	private int m_dimensions;

	private Kernel m_morton_kernel;

	private Kernel m_morton_kernel_group;

	private Context m_context;
	
	public MortonCode(Context context, int dimensions)
	{
		m_context = context;
		m_lookup_table = new Buffer(context, dimensions * dimensions * BYTE_COUNT);
		m_lookup_table.mapBuffer(Buffer.WRITE);
		init(m_lookup_table.handle(), dimensions);
		m_lookup_table.commitBuffer();
		//byte[] check_buffer = BufHelper.rbb(m_lookup_table);
		m_dimensions = dimensions;
		if (context.memoryType() != Context.MAIN_MEMORY)
		{
			if (!context.hasProgram("morton_code_uint32")) 
				create_kernel(context);
			m_morton_kernel = context.getKernel("morton_code_uint32", "morton_code_backup");
			m_morton_kernel_group = context.getKernel("morton_code_uint32", "morton_code_group");
		}
	} 
	
	private void create_kernel(Context context) {
		StringBuffer data = loadKernel("morton.cl");
		context.add("morton_code_uint32",  data.toString());
	}
	
	public void computeMortonCode(Buffer output, Buffer source_points, int num_points)
	{
		if (m_context.memoryType() == Context.MAIN_MEMORY)
		{
			computeMortonCodeCPU(output, source_points, num_points);
			return;
		}
		if (true)
		{
			computeMortonCodeGroup(output, source_points, num_points);
			return;
		}
		
	//	output.fill((byte)0);
		output.checkedFill((byte)0, m_dimensions* DirectMemory.INT_SIZE * num_points);
		
		m_morton_kernel.set_global_size(0,  num_points  );
		//m_morton_kernel.set_local_size(0,  256  );
		
		//m_morton_kernel.set_local_size(1,  1  );
		//m_morton_kernel.set_global_size(1,  4  );
		
		m_morton_kernel.set_arg(0,  output);
		m_morton_kernel.set_arg(1, m_lookup_table);
		m_morton_kernel.set_arg(2, source_points);
		m_morton_kernel.set_arg(3, (int)m_dimensions);
		m_morton_kernel.set_arg(4, (int)num_points);
		m_morton_kernel.invoke();
	
	}
	
	public void computeMortonCodeGroup(Buffer output, Buffer source_points, int num_points)
	{
	//	output.fill((byte)0);
		output.checkedFill((byte)0, m_dimensions* DirectMemory.INT_SIZE * num_points);
		//int wg_size = (int)Math.min(256,  nextPow2(m_dimensions));
		
		m_morton_kernel_group.set_global_size(0,  (int)(num_points * m_dimensions) ); // 1 thread per byte
		//m_morton_kernel_group.set_global_size(0,  (int)(num_points * 256  ) ); // 1 workgroup per point
		//m_morton_kernel_group.set_local_size(0,  256  );
		
	//	m_morton_kernel_group.set_local_size(1,  1  );
	//	m_morton_kernel_group.set_global_size(1,  1  );
		
		m_morton_kernel_group.set_arg(0,  output);
		m_morton_kernel_group.set_arg(1, m_lookup_table);
		m_morton_kernel_group.set_arg(2, source_points);
		m_morton_kernel_group.set_arg(3, (int)m_dimensions);
		m_morton_kernel_group.set_arg(4, (int)num_points);
		
		m_morton_kernel_group.invoke();
		
	}
	
	
	private long nextPow2(long v) {
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v++;
		return v;
	}

	public void computeMortonCodeCPU(Buffer output, Buffer source_points, int num_points)
	{
		if (true)
		{
			computeMortonCodeCPU_pos(output, source_points, num_points);
			return;
		}
		byte[] lookup = BufHelper.rbb(m_lookup_table);
		int[] points = BufHelper.rbi(source_points);
		
		int dims = m_dimensions;
		
		int code_length = dims * 4-1;
	    int rightshift = 32;
	    int total_len = dims * 4;
	    
	    byte[] result = new byte[total_len* num_points];
	    
	    for (int id = 0; id < num_points ; ++id)
	    {
			int input_offset =  id * dims;
			int output_offset = id *total_len;

			for (int i_depth = 1; i_depth <= 4; ++i_depth)
			{ 
				int result_offset = 3 * dims - (i_depth - 1)*dims; // each mask has [dimensions] bytes in it
				for (int pos = result_offset + dims-1;  pos >= result_offset ; --pos)
				{
					for (int d = 0; d < dims; ++d)
					{
						int b = (points[input_offset+d] >> (rightshift - 8 * i_depth)) & 0xFF;
						int offset = d* dims * BYTE_COUNT + b * dims + pos - result_offset;
						result[output_offset + code_length - pos] |= lookup[offset];
						//System.out.println(pos + " i_depth " +  i_depth + " dim "+ d + " byte "+b + "point" + input_offset + " offset "+offset);
					}
				}
			}
	    }
	    
	    output.mapBuffer(Buffer.WRITE);
	    output.writeArray(0, result);
	    output.commitBuffer();
	}
	
	
	public void computeMortonCodeCPU_pos(Buffer output, Buffer source_points, int num_points)
	{
		byte[] lookup = BufHelper.rbb(m_lookup_table);
		int[] points = BufHelper.rbi(source_points);
		
		int dims = m_dimensions;
		
		int code_length = dims * 4-1;
	    int rightshift = 32;
	    int total_len = dims * 32 / 8;
	    
	    int[] result = new int[dims*num_points];
	    
	    for (int gid = 0; gid < num_points * dims ; ++gid)
	    {
	    	
	    	int id = gid * 4;
	    	int update_count = dims / 8;
	    	byte[] cache = new byte[4];
	    	int cache_pos = 0;
	    	for (; cache_pos < 4; ++id, ++cache_pos)
	    	{
	    		int top_dim = update_count ==0 ? dims : dims - 8*(id % update_count);
	    		int low_dim = update_count == 0 ? 0 : top_dim -8;
	    		
				int input_offset =  id /(dims*  4);
				input_offset *= dims;
				int pos = id % (dims *4);
				int i_depth = 1  + pos/dims;
				
				int code_pos = pos % dims; // current offset in dims code.
				for (int d = low_dim; d < top_dim; ++d)
				{
					int b = (points[input_offset+d] >> (rightshift - 8 * i_depth)) & 0xFF;
					int offset = d* dims * BYTE_COUNT + b * dims + dims-1 - code_pos;
					cache[cache_pos] |= lookup[offset];
				//	System.out.println(pos + " i_depth " +  i_depth + " dim "+ d + " byte "+b + "point" + input_offset + " offset "+offset + " value "+ lookup[offset]);
				}
	    	}
	    	//for (int i = 0;i < cache.length ; ++i)
	    	//	if (cache[i]!=-1)
	    	//		System.out.println("break");
	    	result[gid] = cache[3] << 24;
	    	result[gid]|= cache[2] << 16;
	    	result[gid]|= cache[1] << 8;
	    	result[gid]|= cache[0];
	    	
	    }
	    
	    output.mapBuffer(Buffer.WRITE);
	    output.writeArray(0, result);
	    output.commitBuffer();
	}

	
	

	/** native initialization of the lookup table buffer */
	private native void init(long lookupTableHandle, int dims);

	public Buffer lookupTable() {
		return m_lookup_table;
	}
}

