package org.moa.gpu;

import org.moa.gpu.SparseInstanceBuffer.Kind;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

public class SparseMatrix {
	
	protected Buffer m_column_data;
	private int m_columns;
	
	protected Context m_context;
	
	 protected Buffer m_elements;

	private int m_entries_in_batch;
	protected Kind m_kind;
	protected boolean m_mapped;
	protected int m_mode;
	/** 
	 * currently allocated size. 
	 */
	protected int m_number_of_elements;
	/** 
	 * position in the row jumper buffer
	 */
	protected int m_row;
	protected int m_row_block_num;
	private Buffer m_row_blocks;
	protected Buffer m_row_jumper;
	/** 
	 * position in the elements buffer
	 */
	protected int m_row_position;
	protected long m_total_rows;
	protected long m_value_size;
	public SparseMatrix(Kind kind, Context ctx, int rows, int columns, int total_elements)
	{
		m_kind = kind;
		m_context = ctx;
		m_columns = columns;
		m_number_of_elements = total_elements;
		if (kind == Kind.DOUBLE_BUFFER)
			m_value_size = DirectMemory.DOUBLE_SIZE;
		else
		if (kind == Kind.FLOAT_BUFFER)
			m_value_size = DirectMemory.FLOAT_SIZE;
		m_total_rows = rows;
        m_row_jumper = new Buffer(m_context, (rows+1) * DirectMemory.INT_SIZE, Buffer.READ_WRITE);
        m_row_blocks = new Buffer(m_context, (rows+1) * DirectMemory.INT_SIZE, Buffer.READ_WRITE); // 1 block per row... resize if needed.
        m_mapped = false;
        m_row_block_num = 0;
        m_entries_in_batch = 0;
        reallocateSparseMatrix(m_context, m_number_of_elements);
	}
	/** 
     * prepare memory for operation with Buffer.MODE
     * @param mode
     */
    public void begin(int mode)
    {
    	if (m_mapped)
    	{
    		if (mode == m_mode)
    			return;
    		throw new RuntimeException("begin()");
    	}
    	m_row_blocks.mapBuffer(Buffer.WRITE);
    	m_elements.mapBuffer(mode);
    	m_column_data.mapBuffer(mode);
    	m_row_jumper.mapBuffer(mode);
    	m_mode = mode;
    	m_mapped = true;
    }
	public void commit()
    {
    	if (!m_mapped)
    		return;
    	if (m_entries_in_batch > 0)
    		m_row_blocks.writeInt( (++m_row_block_num)*DirectMemory.INT_SIZE, m_row);
    	m_row_blocks.commitBuffer();
    	m_elements.commitBuffer();
    	m_column_data.commitBuffer();
    	m_row_jumper.commitBuffer();
    	m_mapped = false;
    }
	public int getColumnCount() {
		return m_columns;
	}
	public Buffer getColumnIndices() {
		return m_column_data;
	}
	public Buffer getElements() {
		return m_elements;
	}

	public Kind getKind() {
		return m_kind;
	}

	public int getRowBlockNum()
	{
		return m_row_block_num;
	}

	public Buffer getRowBlocks() {
		return m_row_blocks;
	}

	public Buffer getRowJumper() {
		return m_row_jumper;
	}
	
	
	public int getRowPostion() {
		// TODO Auto-generated method stub
		return m_row_position;
	}
	
	/**
	 * Reallocate sparse portion of data
	 * @param context
	 * @param number_of_elements
	 */
	protected void reallocateSparseMatrix(Context context, int number_of_elements) {
		long byte_size = number_of_elements * m_value_size;
	    m_elements = new Buffer(context, byte_size, Buffer.READ_WRITE);
	    m_column_data = new Buffer(context, number_of_elements* DirectMemory.INT_SIZE);
	}
	protected void resize() {
		int new_size = (int)(m_number_of_elements * 1.5);
		Buffer old_elements = m_elements;
		Buffer old_columns = m_column_data;
		
		old_elements.commitBuffer();
		old_columns.commitBuffer();
		
		// update indices here
		reallocateSparseMatrix(m_context, new_size);
		m_number_of_elements = new_size;
		old_elements.copyTo(m_elements);
		old_columns.copyTo(m_column_data);
		m_column_data.mapBuffer(Buffer.WRITE);
		m_elements.mapBuffer(Buffer.WRITE);
		
		// release immediately to prevent memory peaking out
		old_elements.release();
		old_columns.release();
	}
	protected void updateRowBlockBuffer(int entries_in_row)
	{
		if (m_row_block_num == 0)
			m_row_blocks.writeInt(DirectMemory.INT_SIZE * m_row_block_num, 0); // first block
		int shared_memory_size = 1024;
		if (entries_in_row/1024 + m_row_block_num >= m_row_blocks.byteSize()/DirectMemory.INT_SIZE)
		{
			
			Buffer old_row_blocks = m_row_blocks;
			old_row_blocks.commitBuffer();
			
			m_row_blocks = new Buffer(m_context, (int)(m_row_blocks.byteSize()*1.5));
			old_row_blocks.copyTo(m_row_blocks);
						
			m_row_blocks.mapBuffer(Buffer.WRITE);
		}
		for (int i = m_row; i <= m_row ; ++i) 
		{
			m_entries_in_batch += entries_in_row;;
			if (m_entries_in_batch > shared_memory_size)
			{
				int rows_in_batch = i - m_row_blocks.readInt(m_row_block_num*DirectMemory.INT_SIZE);
				if (rows_in_batch > 0) // at least one full row is in the batch. Use current row in next batch.
					m_row_blocks.writeInt((++m_row_block_num)*DirectMemory.INT_SIZE, i--);
				else // row is larger than buffer in shared memory
					m_row_blocks.writeInt((++m_row_block_num)*DirectMemory.INT_SIZE, i+1);
					m_entries_in_batch = 0;
			}
		}
		
	}
	public void reset() {
        m_row_block_num = 0;
        m_entries_in_batch = 0;
        m_row = 0;
		
	}
	
}
