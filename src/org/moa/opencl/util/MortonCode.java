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
	
	public MortonCode(Context context, int dimensions)
	{
		m_lookup_table = new Buffer(context, dimensions * dimensions * BYTE_COUNT);
		m_lookup_table.mapBuffer(Buffer.WRITE);
		init(m_lookup_table.handle(), dimensions);
		m_lookup_table.commitBuffer();
		m_dimensions = dimensions;
		
		if (!context.hasProgram("morton_code_uint32")) 
			create_kernel(context);
		m_morton_kernel = context.getKernel("morton_code_uint32", "morton_code");
		m_morton_kernel_group = context.getKernel("morton_code_uint32", "morton_code_group");
	}
	
	private void create_kernel(Context context) {
		StringBuffer data = loadKernel("morton.cl");
		context.add("morton_code_uint32",  data.toString());
	}
	
	public void computeMortonCode(Buffer output, Buffer source_points, int num_points)
	{
		if (m_dimensions > 256)
		{
			computeMortonCodeGroup(output, source_points, num_points);
			return;
		}
	//	output.fill((byte)0);
		output.checkedFill((byte)0, m_dimensions* DirectMemory.INT_SIZE * num_points);
		
		m_morton_kernel.set_global_size(0,  256*40  );
		m_morton_kernel.set_local_size(0,  256  );
		
	//	m_morton_kernel.set_local_size(1,  1  );
	//	m_morton_kernel.set_global_size(1,  4  );
		
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
		
		m_morton_kernel_group.set_global_size(0,  256*num_points  );
		m_morton_kernel_group.set_local_size(0,  256  );
		
	//	m_morton_kernel_group.set_local_size(1,  1  );
	//	m_morton_kernel_group.set_global_size(1,  1  );
		
		m_morton_kernel_group.set_arg(0,  output);
		m_morton_kernel_group.set_arg(1, m_lookup_table);
		m_morton_kernel_group.set_arg(2, source_points);
		m_morton_kernel_group.set_arg(3, (int)m_dimensions);
		m_morton_kernel_group.set_arg(4, (int)num_points);
		
		m_morton_kernel_group.invoke();
	}
	
	public void computeMortonCodeCPU(Buffer output, Buffer source_points, int num_points)
	{
		if (true)
		throw new RuntimeException("Bug in CPU code");
		
		output.mapBuffer(Buffer.READ);
		source_points.mapBuffer(Buffer.READ);
		m_lookup_table.mapBuffer(Buffer.READ);

		int rightshift = 32;
		int total_len = m_dimensions * 32 / 8;
		for (int id = 0; id < num_points; ++id)
		{
			int input_offset = id*m_dimensions;
			int output_offset = id * total_len;

			for (int i_depth = 1; i_depth <= 4; ++i_depth)
			{
				int result_offset = 3 * m_dimensions - (i_depth - 1)*m_dimensions; // each mask has [dimensions] bytes in it
				for (int pos = result_offset; pos < result_offset + m_dimensions; ++pos)
				{
					for ( int d = 0; d < m_dimensions; ++d)
					{
						
						int byte_ = (source_points.readInt(input_offset + d) >> (rightshift - 8 * i_depth)) & 0xFF;
						int offset = d* m_dimensions * BYTE_COUNT + byte_ * m_dimensions + pos - result_offset;
						byte res = (byte) (output.readByte(output_offset + pos) | m_lookup_table.readByte(offset));
						output.writeByte(output_offset + pos, res);
						
					}
				}
			}
		}
		output.commitBuffer();
		source_points.commitBuffer();
		m_lookup_table.commitBuffer();
	}
	
	

	/** native initialization of the lookup table buffer */
	private native void init(long lookupTableHandle, int dims);

	public Buffer lookupTable() {
		return m_lookup_table;
	}
}

