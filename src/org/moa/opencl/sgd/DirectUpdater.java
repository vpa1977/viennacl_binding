package org.moa.opencl.sgd;

import org.moa.opencl.util.BufHelper;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.MappedFile;

public class DirectUpdater {
	private Context m_context;
	private double m_learning_rate;
	private double m_decay_rate;
	private int m_num_attributes;
	private int m_num_classes;
	private int m_step;
	private double[] m_weights;
	private int m_class_index;
	
	private MappedFile m_shared_weights;
	
	public DirectUpdater(Context ctx, int num_attributes, int num_classes, int class_index)
	{
		m_context = ctx;
		m_learning_rate = 0.0001;
		m_decay_rate = 0.0001;
		m_num_classes = num_classes;
		m_num_attributes = num_attributes;
		m_weights = new double[m_num_classes * m_num_attributes];
		try {
			m_shared_weights = new MappedFile("weights.txt", m_num_classes * m_num_attributes * DirectMemory.DOUBLE_SIZE );
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		m_class_index = class_index;
		m_step = 1;
		
	}
	
	

	public synchronized  void applyUpdate(Buffer gradient_buffer, int batch_number)
	{
		if (true)
		{
			gradient_buffer.mapBuffer(Buffer.READ);
			nativeAtomicUpdate(gradient_buffer.handle(), m_shared_weights.getAddr(), m_num_attributes * m_num_classes, m_learning_rate);
			gradient_buffer.commitBuffer();
			return;
		}
		double[] gradients = BufHelper.rb(gradient_buffer);
		for (int i = 0; i < m_num_classes ; ++i)
		{
			for (int j = 0; j < m_num_attributes; ++j)
			{
				//applyDecay(i, j);
				m_weights[i * m_num_attributes + j] += m_learning_rate * gradients[i * m_num_attributes + j];
			}
		}
		DirectMemory.writeArray(m_shared_weights.getAddr(), 0, m_weights);
	}

	private native void nativeAtomicUpdate(long gradient_buffer, long shared_addr, int size, double learning_rate);
	



	private void applyDecay(int class_index, int att_index) {
		if (att_index != m_class_index)
			m_weights[class_index * m_num_attributes + att_index] *= (1 - (m_decay_rate* m_learning_rate)/(m_step++));
	}
	public synchronized void readWeights(Buffer weights)
	{
		if (true)
		{
			readSharedWeights(weights);
			return;
		}
		weights.mapBuffer(Buffer.WRITE);
		weights.writeArray(0, m_weights);
		weights.commitBuffer();
	}
	
	private void readSharedWeights(Buffer weights) {
		weights.mapBuffer(Buffer.WRITE);
		DirectMemory.writeRaw(m_shared_weights.getAddr(), weights.handle(), weights.byteSize());
		weights.commitBuffer();
	//	BufHelper.print("weights" , weights,(int)( weights.byteSize()/DirectMemory.DOUBLE_SIZE));
	}



	public void updateTau() 
	{
		
	}
	
	public void applyWeightsDelta()
	{
	}
	
	public double[] getWeights()
	{
		return  m_weights;
	}
	

	public double[] getBias() {
		double [] b = new double[m_num_classes];
		for (int i = 0;i < m_num_classes; ++i)
			b[i] = m_weights[ m_num_attributes * i + m_class_index];
		return b;
	}
}
