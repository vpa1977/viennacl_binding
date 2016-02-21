/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.moa.gpu;


import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author john
 */
public class SlidingWindow {

	/** 
	 * true if sliding window write is in progress
	 */
	private boolean m_is_writing;
    /**
     * class data
     */
    private DenseInstanceBuffer m_model;
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
    
    public SlidingWindow(DenseInstanceBuffer.Kind kind, Context ctx, Instances dataset,  int size) {
    	m_is_writing = false;
        m_size = size;
        m_dataset = dataset;
        m_ready = false;
        m_row = 0;
        m_model = new DenseInstanceBuffer(kind,ctx, m_size, m_dataset.numAttributes());
        m_input_data = new Instance[size];
    }

    public void update(Instance instance) {
        m_model.set(instance, m_row);
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
    
    public DenseInstanceBuffer model() {
        return m_model;
    }

    public boolean isReady() {
        return m_ready;
    }

    public void dispose() {
        m_ready = false;
        m_model = null;
    }

	public void begin() {
		if (m_is_writing)
			return;
		m_is_writing = true;
		m_model.begin(Buffer.WRITE);
	}

	public void commit() {
		if (m_is_writing)
		{
			m_model.commit();
		}
		m_is_writing = false;
	}

	public void beginRead() {
		m_model.begin(Buffer.READ);
	}

	public Instance read(int pos) {
		DenseInstance flyweight = m_model.read(pos, m_dataset);
		return flyweight;
	}

	public double weight(int i) {
		return m_input_data[i].weight();
	}

	public double classValue(int i) {
		return m_input_data[i].classValue();
	}

}
