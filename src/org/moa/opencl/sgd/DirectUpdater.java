package org.moa.opencl.sgd;

import java.io.File;
import org.moa.opencl.util.AbstractUtil;

import org.moa.opencl.util.BufHelper;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.MappedFile;

public class DirectUpdater extends AbstractUtil implements Updater {
	private Context m_context;
	private double m_learning_rate;
	private double m_decay_rate;
	private int m_num_attributes;
	private int m_num_classes;
	private int m_step;
	private double[] m_weights;
	private int m_class_index;
	
	private MappedFile m_shared_weights;
  private SharedSemaphores m_semaphore;
  private final int m_total_workers;
  private final int m_id;
	
	public DirectUpdater(Context ctx, int num_attributes, int num_classes, int class_index, int id, int num_workers)
	{
		m_context = ctx;
		m_learning_rate = 0.0001;
		m_decay_rate = 0.0001;
		m_num_classes = num_classes;
		m_num_attributes = num_attributes;
		m_weights = new double[m_num_classes * m_num_attributes];
    
		try {
			if (id == 0)
      {
				new File("weights.txt").delete();
        new File("semaphore.txt").delete();
      }
			m_shared_weights = new MappedFile("weights.txt", m_num_classes * m_num_attributes * DirectMemory.DOUBLE_SIZE );
      m_semaphore = new SharedSemaphores("semaphore.txt", num_workers);
      if (id == 0)
      {
        for (int i = 0; i < num_workers; ++i)
          m_semaphore.setValue(i, false);
      }
      
      for (int i = 0; i < num_workers; ++i)
          System.out.println("Initial semaphore state " + m_semaphore.getValue(i) );

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    m_total_workers = num_workers;
		m_class_index = class_index;
		m_step = 1;
    m_id = id;
		
    if (!ctx.hasProgram("1bitsgd_update") && ctx.memoryType() == Context.HSA_MEMORY) 
		{
			StringBuffer program = new StringBuffer();
			program.append("#define VALUE_TYPE "+ type() + "\n");
			program.append("#define COND_TYPE "+ cond_type() + "\n");
			program.append(loadKernel("one_bit_updaters.cl"));
			ctx.add("1bitsgd_update", program.toString());
		}
	}
  
    
  public String type() 
  {
    return "double";
  }
	
  public String cond_type() 
  {
    return "long";
  }

	
  private void waitIdle() {
		boolean free;
		do {
			free = true;
			for (int i = 0; i < m_total_workers; ++i)
			{
				if (!m_semaphore.getValue(i))
					free = false;
			}
		} while (!free);
	}
	
  public static boolean use_native = true;
	

	public synchronized  void applyUpdate(Buffer gradient_buffer, int batch_number)
	{
    m_semaphore.setValue(m_id, true);
    waitIdle();
		if (use_native)
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
				applyDecay(i, j);
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
		if (use_native)
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
    double[] weights = getWeights();
		double [] b = new double[m_num_classes];
		for (int i = 0;i < m_num_classes; ++i)
			b[i] = weights[ m_num_attributes * i + m_class_index];
		return b;
	}
}
