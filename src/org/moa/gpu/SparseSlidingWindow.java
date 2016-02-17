package org.moa.gpu;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;

import weka.core.Instance;
import weka.core.Instances;
/**
 * Батчи одноразовые. 10+ батчей по 1024 скажем в батче. 
 * Кормятся на вход SGD. 
 * Или онлайн SGD - там все по очереди. 
 *  
 * @author john
 *
 */

public class SparseSlidingWindow {
	/** 
	 * true if sliding window write is in progress
	 */
	private boolean m_is_writing;
    /**
     * class data
     */
    private SparseInstanceBuffer[] m_models;
    /**
     * number of instances in the window
     */
    private int m_size;
    /**
     * current row in the window
     */
    private int m_row;
    /**
     * return true if enough instances present
     */
    private boolean m_ready;
    
    private Instances m_dataset;
    
    private Instance[] m_input_data;
	private int m_minibatch_size;
	private Context m_context;
    
    public SparseSlidingWindow(Context ctx, Instances dataset,  int size, int minibatch_size) {
    	m_is_writing = false;
        m_size = size;
        m_dataset = dataset;
        m_ready = false;
        m_row = 0;
        m_minibatch_size = minibatch_size;
        m_models = new SparseInstanceBuffer[size];
        m_input_data = new Instance[size];
        m_context = ctx;
        
    }

    public void update(Instance instance) {
    	int index = m_row / m_minibatch_size;
    	SparseInstanceBuffer  buffer = m_models[index];
    	
    	if (buffer.rows() == m_minibatch_size)
    		buffer = new SparseInstanceBuffer(m_context,m_minibatch_size, m_dataset.numAttributes(), 0.2 );
    	buffer.begin(Buffer.WRITE);
        buffer.append(instance);
        m_input_data[m_row] = instance;
        m_row++;
        if (m_row == m_size) {
            m_ready = true;
            m_row = 0;
        }
    }

    
    public Instances dataset()
    {
    	return m_dataset;
    }
    
    public SparseMatrix[] models() {
        return m_models;
    }

    public boolean isReady() {
        return m_ready;
    }

    public void dispose() {
        m_ready = false;
        m_models = null;
    }

	public void begin() {
		if (m_is_writing)
			return;
		m_is_writing = true;
	}

	public void commit() {
		
		if (m_is_writing)
		{
			for (SparseInstanceBuffer b : m_models)
				b.commit();
		}
		m_is_writing = false;
	}


	public double weight(int i) {
		return m_input_data[i].weight();
	}

	public double classValue(int i) {
		return m_input_data[i].classValue();
	}
}
