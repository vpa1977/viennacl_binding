package moa.classifiers.gpu.zorder;

/** 
 * ZOrderedItem contains 
 * link to zorder data array, 
 * position in the array, 
 * number of dimensions for morton code, 
 * and instance index in the sliding window 
 * 
 * @author john
 *
 */
public class ZOrderItem implements Comparable<ZOrderItem>{
	/** 
	 * position in the m_data (actual index is m_pos * m_dims )
	 */
	private int m_pos;
	/** 
	 * ZOrder code data. each dimension is represented by 32 bits (1 int)
	 */
	private byte[] m_data;
	/** 
	 *  number of dimensions per code 
	 */  
	private int m_size;
	/** 
	 * instance index in the sliding window
	 */
	private int m_instance_index;

	public ZOrderItem(byte[] data, int pos, int instance_index, int dims)
	{
		m_pos = pos;
		m_data = data;
		m_size = dims;
		m_instance_index = instance_index;
	}
	
	public boolean equals(Object o ) 
	{
		if (!(o instanceof ZOrderItem))
			return false;
		if (o  == null)
			return false;
		ZOrderItem item = (ZOrderItem)o;
		if (item.m_size != m_size)
			return false;
		for (int i = 0; i< m_size ; ++i)
			if (m_data[m_pos + i] != item.m_data[item.m_pos + i])
				return false;
		return true;
	}

	@Override
	public int compareTo(ZOrderItem o) {
		assert(m_size == o.m_size);
		int this_offset = m_pos;
		int other_offset =o.m_pos;
		for (int i = m_size-1; i >=0; --i)
		{
			int x = m_data[this_offset + i] & 0xFF;
			int y = o.m_data[other_offset+i]& 0xFF;
			if (x > y)
				return 1;
			if (x < y)
				return -1;
		}
		return 0;
	}



	public void print() {
		for (int i = 0; i < m_size ; ++i)
			System.out.print((m_data[m_pos+i]  & 0xFF) + " ");
		System.out.println();
		
	}

	public byte[] code() {
		byte[] c = new byte[m_size];
		System.arraycopy(m_data, m_pos, c, 0, m_size);
		return c;
	}

	public int offset() {
		// TODO Auto-generated method stub
		return m_pos;
	}

	public int instanceIndex() {
		// TODO Auto-generated method stub
		return m_instance_index;
	}

}
