package org.moa.opencl.sgd;

import java.sql.Time;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.TimeUnit;

import javax.rmi.CORBA.Util;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.SparseInstanceBuffer;
import org.moa.gpu.UnitOfWork;
import org.moa.opencl.util.BufHelper;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;

import moa.classifiers.functions.SGDMultiClass;
import moa.core.DoubleVector;
import weka.core.Instance;
import weka.core.Instances;

public class HogwildScheme {
	private SimpleUpdater m_updater;
	private Multinominal[] m_gradient;
	private Buffer[] m_weights; // per minibatch weight copies;
	private Buffer m_local_weights;
	private Multinominal m_predictor;
	private Buffer[] m_bias;
	private ForkJoinPool m_pool;
	private ArrayBlockingQueue<UnitOfWork> m_work;
	private ArrayBlockingQueue<UnitOfWork> m_progress;
	private int m_current_batch;
	private int m_num_batches;
	private Instances m_dataset;
	private Context m_context;
	private int m_minibatch_size;
	private DenseInstanceBuffer m_test_instance;


	public HogwildScheme(Context ctx, Instances dataset, int num_batches, int minibatch_size) {
		m_updater = new SimpleUpdater(ctx, dataset.numAttributes(), dataset.numClasses(), num_batches);
		m_gradient = new Multinominal[num_batches];
		m_weights = new Buffer[num_batches];
		m_bias = new Buffer[num_batches];
		m_predictor = new Multinominal(ctx, dataset.numClasses(), dataset.numAttributes(), minibatch_size);
		m_local_weights = new Buffer(ctx, dataset.numClasses() * dataset.numAttributes() * m_predictor.typeSize());
		m_test_instance = new DenseInstanceBuffer(ctx, 1, dataset.numAttributes(), Buffer.READ);
		m_test_instance.setClassReplaceValue(1);
		for (int i = 0; i < num_batches; ++i) {
			m_gradient[i] = new Multinominal(ctx, dataset.numClasses(), dataset.numAttributes(), minibatch_size);
			m_weights[i] = new Buffer(ctx, dataset.numClasses() * dataset.numAttributes() * m_gradient[i].typeSize());
			m_weights[i].fill((byte)0);
			m_bias[i] = new Buffer(ctx, num_batches * dataset.numClasses() * m_gradient[i].typeSize());
			m_bias[i].fill((byte)0);
		}

		m_pool = new ForkJoinPool(num_batches);
		m_work = new ArrayBlockingQueue<UnitOfWork>(2*num_batches);
		m_progress = new ArrayBlockingQueue<UnitOfWork>(2*num_batches);
		m_current_batch = 0;
		m_num_batches = num_batches;
		m_dataset = dataset;
		m_context = ctx;
		m_minibatch_size = minibatch_size;
		new Thread(m_TrainThread).start();
	}
	
	private Runnable m_TrainThread = new Runnable() 
	{
		@Override
		public void run() {
			while (true)
			{
				UnitOfWork work = null;
				try {
					work = m_progress.take();
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

				if (m_current_batch == m_num_batches) {
					updateWeightsAndTau();
					m_current_batch = 0;
				}
				int batch = m_current_batch ++;
				m_updater.readWeights(m_weights[batch]);
				if (work instanceof DenseInstanceBuffer) {
					DenseInstanceBuffer dib = (DenseInstanceBuffer) work;
					m_gradient[batch].computeGradient(m_dataset, dib, m_weights[batch], m_bias[batch]);
					Buffer b = m_gradient[batch].getComputedGradients();
				//	double[] gradients = BufHelper.rb(b);
				//	double[] weights = BufHelper.rb(m_weights[batch]);
				//	int[] wd = BufHelper.rbi(m_updater.m_weights_delta);
				//	double[] tau = BufHelper.rb(m_updater.m_tau);
					m_updater.applyUpdate(m_gradient[batch].getComputedGradients(), batch);
					
					//gradients = BufHelper.rb(b);
					//weights = BufHelper.rb(m_weights[batch]);
					//wd = BufHelper.rbi(m_updater.m_weights_delta);
					//tau = BufHelper.rb(m_updater.m_tau);
					//System.out.println();
				} else if (work instanceof SparseInstanceBuffer) {
					throw new RuntimeException("Not supported");
					//SparseInstanceBuffer dib = (SparseInstanceBuffer) work;
					//m_gradient[batch].computeGradient(m_dataset, dib, m_weights[batch], m_bias[batch]);
					//m_updater.applyUpdate(m_gradient[batch].getComputedGradients(), batch);
				}
				try {
					m_work.put(work);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
			}
			
		}

	};

	public synchronized Buffer getWeights() {
		Buffer copy = new Buffer(m_context, m_updater.getWeights().byteSize());
		updateWeightsAndTau();
		m_updater.getWeights().copyTo(copy);
		return copy;
	}

	public synchronized void put(final UnitOfWork work) {
		final int batch = m_current_batch;

		
		ForkJoinTask<Boolean> task = new ForkJoinTask<Boolean>() {

			@Override
			public Boolean getRawResult() {
				// TODO Auto-generated method stub
				return true;
			}

			@Override
			protected void setRawResult(Boolean value) {
				// TODO Auto-generated method stub

			}
			
			public void doit()
			{
				exec();
			}

			@Override
			protected boolean exec() {
				try {
					work.commit();
					m_progress.put(work);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				return true;
			}
		};

		m_pool.submit(task);
		//task.invoke();
	}

	/**
	 * update global model parameters
	 */
	public void updateWeightsAndTau() {
		m_pool.awaitQuiescence(Long.MAX_VALUE, TimeUnit.DAYS);
		m_updater.applyWeightsDelta();
		m_updater.updateTau();
	}

	public UnitOfWork take() {
		try {
			return m_work.take();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null; // should not happen
	}

	public void populate(boolean dense) {
		int capacity = m_work.remainingCapacity();
		if (dense) {
			for (int i = 0; i < capacity; ++i)
			{
				DenseInstanceBuffer b = new DenseInstanceBuffer(m_context, m_minibatch_size, m_dataset.numAttributes(), Buffer.READ);
				b.setClassReplaceValue(1);
			
				
				m_work.add(b);
			}
		} else {
			for (int i = 0; i < capacity; ++i)
				m_work.add(new SparseInstanceBuffer(m_context, m_minibatch_size, m_dataset.numAttributes(), 0.1));
		}
	}

	public double[] getVotesForInstance(Instance inst) {
		m_updater.readWeights(m_local_weights);
		m_test_instance.begin(Buffer.WRITE);
		m_test_instance.set(inst, 0);
		m_test_instance.commit();
	//	double[] weights = BufHelper.rb(m_local_weights);
/*for (double d : weights)
			System.out.print(d+  " ");
		System.out.println();*/
		return m_predictor.predict(inst.dataset(), m_test_instance, m_local_weights, m_bias[0]);
	}

	public Buffer getErrorLarge() 
	{
		return m_updater.getErrorLarge();
	}

	public Buffer getErrorSmall() 
	{
		return m_updater.getErrorSmall();
	}
	
	public Buffer getTau() {
		return m_updater.getTau();
	}
}
