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
	//private DirectUpdater m_updater;
	private OneBitUpdater m_updater;
	private Multinominal m_gradient;
	private Buffer m_weights; // per minibatch weight copies;
	private Buffer m_local_weights;
	private Instances m_dataset;
	private Context m_context;
	
	private DenseInstanceBuffer m_test_instance;
	private int m_self_id; /* identifier of the worker. worker 0 is responsible for weights update */
	private Buffer m_bias;

 
	public HogwildScheme(Context ctx, Instances dataset, int self_id, int minibatch_size) {
		//m_updater = new DirectUpdater(ctx, dataset.numAttributes(), dataset.numClasses(), dataset.classIndex());
		//m_updater = new SimpleUpdater(ctx, dataset.numAttributes(), dataset.numClasses(), dataset.classIndex(), 100);
		m_updater = new OneBitUpdater(ctx, dataset.numAttributes(), dataset.numClasses(), dataset.classIndex(), 1, 0, 10);
		m_test_instance = new DenseInstanceBuffer(ctx, 1, dataset.numAttributes(), Buffer.READ);
		m_test_instance.setClassReplaceValue(1);
		m_gradient = new Multinominal(ctx, dataset.numClasses(), dataset.numAttributes(), minibatch_size);
		m_local_weights = new Buffer(ctx, dataset.numClasses() * dataset.numAttributes() * m_gradient.typeSize());

		m_weights = new Buffer(ctx, dataset.numClasses() * dataset.numAttributes() * m_gradient.typeSize());
		m_weights.fill((byte)0);
		m_self_id = self_id;
		m_dataset = dataset;
		m_context = ctx;
	}
	
	public int getId()
	{
		return m_self_id;
	}
	
	public void trainStep(UnitOfWork work , int step)
	{
			
		if (work instanceof DenseInstanceBuffer)
		{
			DenseInstanceBuffer dib = (DenseInstanceBuffer) work;
			m_updater.readWeights(m_weights);
			m_gradient.computeGradient(m_dataset, dib, m_weights);
			m_updater.applyUpdate(m_gradient.getComputedGradients(), 0);
		}
		else
		{
			SparseInstanceBuffer dib = (SparseInstanceBuffer) work;
			m_updater.readWeights(m_weights);
			m_gradient.computeGradient(m_dataset, dib, m_weights);
			m_updater.applyUpdate(m_gradient.getComputedGradients(), 0);
		}
	}
	

	public synchronized double[] getWeights() {
		return m_updater.getWeights();
	}

	/**
	 * update global model parameters
	 */
	public void updateWeightsAndTau() {
		m_updater.applyWeightsDelta();
		m_updater.updateTau();
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
		return m_gradient.predict(inst.dataset(), m_test_instance, m_local_weights);
	}

	public Buffer getErrorLarge() 
	{
		return null;//m_updater.getErrorLarge();
	}

	public Buffer getErrorSmall() 
	{
		return null;//return m_updater.getErrorSmall();
	}
	
	public Buffer getTau() {
		return null;//return m_updater.getTau();
	}

	public double[] getBias() {
		return m_updater.getBias();
	}
}
