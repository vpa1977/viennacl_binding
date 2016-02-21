package org.moa.gpu;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class SparseInstanceBuffer extends SparseMatrix implements UnitOfWork {
	
	public enum Kind
	{
		DOUBLE_BUFFER, 
		FIXED_INT_BUFFER, 
		FLOAT_BUFFER
	}

    
    private class SparseInstanceAccess extends SparseInstance {
        public SparseInstanceAccess(Instance inst) {
            super(inst);
        }
        
        public int[] indices(){
        	return m_Indices;
        }
        

        public double[] values() {
            return m_AttValues;
        }
    }
    
    private Buffer m_class_buffer;
    private long m_number_of_attributes;
    private Buffer m_weights;
    
    public void reset()
    {
    	m_row_position = 0;
    	super.reset();
    }
    
    public SparseInstanceBuffer(SparseInstanceBuffer.Kind kind, Context context, int rows, int numAttributes, double fill_ratio) 
    {
    	super(kind, context, rows,numAttributes ,  (int)(rows * numAttributes  * fill_ratio));
        m_number_of_attributes = numAttributes;
        m_class_buffer =  new Buffer(context, m_total_rows * m_value_size, Buffer.READ_WRITE);
        m_weights = new Buffer(context, m_total_rows * m_value_size, Buffer.READ_WRITE);
        m_row_position = 0;
    }

    public SparseInstanceBuffer(Context context, int rows, int numAttributes, double fill_ratio) {
    	this(Kind.DOUBLE_BUFFER, context, rows, numAttributes, fill_ratio);
    }
    
    public boolean append(Instance iis) {
		if (m_row == m_total_rows) 
			return false;
		SparseInstanceAccess access = new SparseInstanceAccess(iis);
		double[] values = access.values();
		
		int[] indices = access.indices();
		// attempt to compact - align data to the start of the buffer
		// resize
		if (indices.length + m_row_position > m_number_of_elements)
			resize();
		updateRowBlockBuffer(indices.length);	
		int column_position = m_row_position;
		// write last row position to row jumper
		m_row_jumper.writeInt(m_row*DirectMemory.INT_SIZE, column_position); // mark start of row
		m_column_data.writeArray(m_row_position, indices);
		if (m_kind == Kind.FLOAT_BUFFER)
		{
			float[] dup = new float[values.length];
			for (int i = 0; i < values.length; ++i)
				dup[i] = (float)values[i];
			m_elements.writeArray(m_row_position, dup);
		}
		else
		{
			m_elements.writeArray(m_row_position, values);
		}
		
		
		m_class_buffer.write(m_row * m_value_size, iis.classValue()); 
	    m_weights.write(m_row * m_value_size, iis.weight());

	    m_row ++;
	    m_row_position += values.length;
	    return true;
	}

   

    public Buffer classes() {
        return m_class_buffer;
    }


    public double classValueOf(int pos) {
		double classValue = m_class_buffer.read(pos * m_value_size);
        
        return classValue;
	}


	public void commit()
    {
    	if (!m_mapped)
    		return;
    	m_row_jumper.writeInt(m_row*DirectMemory.INT_SIZE, m_row_position); // close row
    	m_class_buffer.commitBuffer();
    	m_weights.commitBuffer();
    	super.commit();
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
    	
    	m_class_buffer.mapBuffer(mode);
    	m_weights.mapBuffer(mode);
    	super.begin(mode);
    }
	
    /** 
     *
     * @return current number of populated rows.
     */
	public int rows() {
		return (int)m_row; 
	}

	public void write(Instances instances) {
		begin(Buffer.WRITE);
		for (int i  = 0; i < instances.size(); ++i)
		{
			Instance next = instances.get(i);
			append(next);
		}
		commit();
	}

	public int getRowPostion() {
		return m_number_of_elements;
	}


}
