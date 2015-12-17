package org.moa.gpu;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class DenseInstanceBuffer {
	
	public enum Kind
	{
		DOUBLE_BUFFER, 
		FLOAT_BUFFER, 
		FIXED_INT_BUFFER
	}

    
    private long m_number_of_attributes;
    
    private long m_rows;
    private Buffer m_class_buffer;
    private Buffer m_attribute_values_buffer;
    private Buffer m_weights;
    private long m_value_size;
    private Kind m_kind;

    public DenseInstanceBuffer(Context context, int rows, int numAttributes) {
    	m_kind = Kind.DOUBLE_BUFFER;
    	m_value_size = DirectMemory.DOUBLE_SIZE;
        m_rows = rows;
        long byte_size = rows * numAttributes * m_value_size;
        m_number_of_attributes = numAttributes;
        m_attribute_values_buffer = new Buffer(context, byte_size, Buffer.READ_WRITE);
        m_class_buffer =  new Buffer(context, m_rows * m_value_size, Buffer.READ_WRITE);
        m_weights = new Buffer(context, m_rows * m_value_size, Buffer.READ_WRITE);
    }
    
    public Kind getKind() 
    {
    	return m_kind;
    }

    public void set(Instance inst, int pos) {
        if (pos >= m_rows) {
            throw new ArrayIndexOutOfBoundsException(pos);
        }
        long writeIndex = pos * m_number_of_attributes;

        DenseInstanceAccess ins = new DenseInstanceAccess(inst);
        double[] data = ins.values();
        int classIndex = inst.classIndex();
        //DirectMemory.writeArray(attribute_handle, writeIndex, data); // write instances
        long offset = (writeIndex + classIndex) * m_value_size;
        m_attribute_values_buffer.writeArray(writeIndex, data);
        m_attribute_values_buffer.write(offset, (double)0);// zero out class attribute
        m_class_buffer.write(pos * m_value_size, inst.classValue()); 
        m_weights.write(pos * m_value_size, inst.weight());
    }

    public DenseInstance read(int pos, Instances dataset) {
        double[] attribute_values = new double[(int)m_number_of_attributes];
        double weightValue = m_weights.read(pos*m_value_size);
        double classValue = m_class_buffer.read(pos * m_value_size);
        m_attribute_values_buffer.readArray(pos * m_number_of_attributes * m_value_size,attribute_values);
        DenseInstance instance = new DenseInstance(weightValue,attribute_values);
        instance.setDataset(dataset);
        instance.setClassValue(classValue);
        return instance;
    }

  

    /** 
     * prepare memory for operation with Buffer.MODE
     * @param mode
     */
    public void begin(int mode)
    {
    	m_class_buffer.mapBuffer(mode);
    	m_attribute_values_buffer.mapBuffer(mode);
    	m_weights.mapBuffer(mode);
    }

    public void commit()
    {
    	m_class_buffer.commitBuffer();
    	m_attribute_values_buffer.commitBuffer();
    	m_weights.commitBuffer();
    }

    public long data() {
        return m_attribute_values_buffer.handle();
    }

    public long classes() {
        return m_class_buffer.handle();
    }


	public double classValueOf(int pos) {
		double classValue = m_class_buffer.read(pos * m_value_size);
        
        return classValue;
	}
	
    private class DenseInstanceAccess extends DenseInstance {
        public DenseInstanceAccess(Instance inst) {
            super(inst);
        }

        public double[] values() {
            return m_AttValues;
        }
    }


	public int rows() {
		return (int)m_rows;
	}

	public Buffer attributes() {
		return m_attribute_values_buffer;
	}



}
