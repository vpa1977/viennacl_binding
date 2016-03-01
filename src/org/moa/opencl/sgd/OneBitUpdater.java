package org.moa.opencl.sgd;

import java.io.File;

import org.moa.opencl.util.BufHelper;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.MappedFile;

import weka.core.tokenizers.WordTokenizer;

public class OneBitUpdater {
	private Context m_context;
	private double m_learning_rate;
	private double m_decay_rate;
	private int m_num_attributes;
	private int m_num_classes;
	private int m_step;
	private double[] m_weights;
	private int m_class_index;
	
	private MappedFile m_shared_weights;
	private MappedFile m_error_large;
	private MappedFile m_error_small;
	private MappedFile m_tau_data;
	private MappedFile m_average_error_large;
	private MappedFile m_average_error_small;
	private SharedSemaphores m_semaphores;
	private long m_value_size = DirectMemory.DOUBLE_SIZE;
	private int m_total_workers;
	private int m_worker_id;
	private int m_total_steps;
	private MappedFile m_weight_delta;
	private int m_update; /* number of tau updates */
	
	
	
	public OneBitUpdater(Context ctx, 
			int num_attributes, 
			int num_classes, 
			int class_index, 
			int total_workers, 
			int worker_id, 
			int total_steps)
	{
		m_context = ctx;
		m_learning_rate = 0.0001;
		m_decay_rate = 0.0001;
		m_num_classes = num_classes;
		m_num_attributes = num_attributes;
		m_weights = new double[m_num_classes * m_num_attributes];
		m_total_workers = total_workers;
		try {
			if (m_worker_id == 0)
			{
				new File("weights.txt").delete();
				new File("weight_delta.txt").delete();
				new File("error_large.txt").delete();
				new File("error_small.txt").delete();
				new File("avg_error_large.txt").delete();
				new File("avg_error_small.txt").delete();
				new File("tau.txt").delete();
				new File("semaphores.txt").delete();

			}
			
			m_shared_weights = new MappedFile("weights.txt", m_num_classes * m_num_attributes * m_value_size );
			m_weight_delta = new MappedFile("weight_delta.txt", m_num_classes * m_num_attributes * DirectMemory.LONG_SIZE );
			
			m_error_large = new MappedFile("error_large.txt", total_workers *m_num_classes * m_num_attributes * m_value_size);
			m_error_small = new MappedFile("error_small.txt", total_workers *m_num_classes * m_num_attributes * m_value_size);

			m_average_error_large = new MappedFile("avg_error_large.txt", total_workers *m_num_classes * m_num_attributes * m_value_size);
			m_average_error_small = new MappedFile("avg_error_small.txt", total_workers *m_num_classes * m_num_attributes * m_value_size);

			m_tau_data = new MappedFile("tau.txt", m_num_classes * m_num_attributes * m_value_size );
			m_semaphores = new SharedSemaphores("semaphores.txt", total_workers);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		m_class_index = class_index;
		m_worker_id = worker_id;
		m_step = 0;
		m_total_steps = total_steps;
		m_update =0;
	}
	
	

	public synchronized  void applyUpdate(Buffer gradient_buffer, int batch_number)
	{
		if (true)
		{
			gradient_buffer.mapBuffer(Buffer.READ);
			nativeComputeUpdate(
					gradient_buffer.handle(), 
					m_shared_weights.getAddr(),
					m_weight_delta.getAddr(),
					m_error_large.getAddr(),
					m_error_small.getAddr(),
					m_tau_data.getAddr(),
					m_num_attributes,
					m_num_classes, 
					m_learning_rate, 
					m_worker_id);
			gradient_buffer.commitBuffer();
			if (m_step >= m_total_steps)
			{
				m_step = 0;
				setIdle();
				if (m_worker_id == 0) /** update tau */ 
				{
					waitIdle();
					updateTau();
					continueWork();
				}
				
			}
			else
			{
				 ++m_step;
			}
			
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



	private void applyDecay(int class_index, int att_index) {
		if (att_index != m_class_index)
			m_weights[class_index * m_num_attributes + att_index] *= (1 - (m_decay_rate* m_learning_rate)/(m_step++));
	}
	
	public  void readWeights(Buffer weights)
	{
		waitWorkAllowance();
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
		nativeComputeWeight(weights.handle(), m_tau_data.getAddr(), m_weight_delta.getAddr(), m_learning_rate, m_num_attributes, m_num_classes);
		weights.commitBuffer();
	//	BufHelper.print("weights" , weights,(int)( weights.byteSize()/DirectMemory.DOUBLE_SIZE));
	}
	
	
	
	private native void nativeComputeUpdate(
			long gradient, 
			long weights, 
			long weights_delta, 
			long error_data_large,
			long error_data_small,
			long tau,
			int num_attributes,
			int num_classes, 
			double learning_rate, 
			int worker_id);
	
	private native void nativeComputeWeight(long weight, long tau, long delta, double rate, int num_attributes, int num_classes);
	private native void nativeUpdateTau(
			long tau, 
			long es_avg, 
			long el_avg, 
			long es, 
			long el, 
			int total_workers,
			int num_attributes, 
			int num_classes, int update_step);


	public void updateTau() 
	{
		nativeComputeWeight(m_shared_weights.getAddr(), 
							m_tau_data.getAddr(), 
							m_weight_delta.getAddr(), 
							m_learning_rate, m_num_attributes, m_num_classes);
		DirectMemory.set(m_weight_delta.getAddr(), m_weight_delta.getSize(),(byte) 0);
		nativeUpdateTau(m_tau_data.getAddr(),
				m_average_error_small.getAddr(),
				m_average_error_large.getAddr(),
				m_error_small.getAddr(),
				m_error_large.getAddr(),
				m_total_workers,
				m_num_attributes,
				m_num_classes,
				++m_update);
			
	}






	private void waitIdle() {
		boolean free;
		do {
			free = true;
			for (int i = 0; i < m_total_workers; ++i)
			{
				if (!m_semaphores.getValue(i))
					free = false;
			}
		} while (!free);
	}
	
	private void setIdle() 
	{
		m_semaphores.setValue(m_worker_id, true);
	}
	
	private void continueWork()
	{
		for (int i = 0; i < m_total_workers; ++i)
		{
			m_semaphores.setValue(i, false);
		}
	}
	
	private void waitWorkAllowance() 
	{
		do 
		{
		} while (m_semaphores.getValue(m_worker_id));
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
