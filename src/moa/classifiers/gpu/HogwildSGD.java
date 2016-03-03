package moa.classifiers.gpu;



import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadPoolExecutor;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.SlidingWindow;
import org.moa.gpu.SparseInstanceBuffer;
import org.moa.gpu.SparseSlidingWindow;
import org.moa.gpu.UnitOfWork;
import org.moa.opencl.knn.DeviceZOrderSearch;
import org.moa.opencl.knn.DoubleLinearSearch;
import org.moa.opencl.knn.Search;
import org.moa.opencl.knn.SimpleZOrderSearch;
import org.moa.opencl.sgd.Gradient;
import org.moa.opencl.sgd.HingeGradient;
import org.moa.opencl.sgd.HogwildScheme;
import org.moa.opencl.sgd.SimpleUpdater;
import org.moa.opencl.sgd.Updater;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.GlobalContext;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.functions.SGDMultiClass;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import moa.tasks.TaskMonitor;
import org.moa.opencl.sgd.DirectUpdater;
import org.moa.opencl.sgd.OneBitUpdater;
import test.org.moa.opencl.IBk;
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

/**
 * Naive implementation backed by the native library.
 *
 * @author bsp
 *
 */
public class HogwildSGD extends AbstractClassifier  {

    
    static
    {
    	System.loadLibrary("viennacl-java-binding");
    }
    
	
	class ReferenceSGD extends SGDMultiClass {
		public DoubleVector[] getWeights() 
		{
			return m_weights;
		}
		
		public double[] getBias() 
		{
			return m_bias;
		}
		
	    @Override
	    public void trainOnInstanceImpl(Instance instance) {

	        if (m_weights == null) {
	            int length;
	             if (instance.classAttribute().isNominal()) {
	                 length = instance.numClasses();
	             } else {
	                 length = 1;
	             }
	            m_weights = new DoubleVector[length];
	            m_bias = new double[length];
	            for (int i = 0; i < m_weights.length; i++){
	                m_weights[i] = new DoubleVector(); 
	                m_bias[i] = 0.0;
	            }
	        }
	      //  System.out.print("wx " );
	        for (int i = 0; i < m_weights.length; i++){
	                this.trainOnInstanceImpl(instance, i); 
	            }
	      //  System.out.println();
	        m_t++;
	    }   
		
	    public void trainOnInstanceImpl(Instance instance, int classLabel) {    
	        if (!instance.classIsMissing()) {

	            double wx = dotProd(instance, m_weights[classLabel], instance.classIndex());
	         //   System.out.print(wx + " ");
	            double y;
	            double z;
	            if (instance.classAttribute().isNominal()) {
	                y = (instance.classValue() != classLabel) ? -1 : 1;
	                z = y * (wx + m_bias[classLabel]);
	            } else {
	                y = instance.classValue();
	                z = y - (wx + m_bias[classLabel]);
	                y = 1;
	            }

	            // Compute multiplier for weight decay
	            double multiplier = 1.0;
	            if (m_numInstances == 0) {
	                multiplier = 1.0 - (m_learningRate * m_lambda) / m_t;
	            } else {
	                multiplier = 1.0 - (m_learningRate * m_lambda) / m_numInstances;
	            }
	            for (int i = 0; i < m_weights[classLabel].numValues(); i++) {
	                m_weights[classLabel].setValue(i,m_weights[classLabel].getValue (i) * multiplier);
	            }

	            // Only need to do the following if the loss is non-zero
	            if (m_loss != HINGE || (z < 1)) {

	                // Compute Factor for updates
	                double factor = m_learningRate * y * dloss(z);

	                // Update coefficients for attributes
	                int n1 = instance.numValues();
	                for (int p1 = 0; p1 < n1; p1++) {
	                    int indS = instance.index(p1);
	                    if (indS != instance.classIndex() && !instance.isMissingSparse(p1)) {
	                        m_weights[classLabel].addToValue(indS, factor * instance.valueSparse(p1));
	                    }
	                }

	                // update the bias
	                m_bias[classLabel] += factor;
	            }
	            
	        }
	    }

	}
	
    private ZeroR m_default_classifier = new ZeroR();
    private transient Context m_context;
    private transient HogwildScheme m_hogwild_scheme;
    private transient ReferenceSGD m_reference_sgd;
    public HogwildSGD()
    {
    	
    }
    
    public IntOption workerIndexOption = new IntOption( "workerIndex", 'i', "Worker Index", 0, 0, Integer.MAX_VALUE);
    public IntOption workerCountOption = new IntOption( "workerCount", 'p', "Worker Count", 1, 1, Integer.MAX_VALUE);
    public IntOption staggerOption = new IntOption( "stagger", 'o', "Stagger", 10, 1, Integer.MAX_VALUE);
    
    public MultiChoiceOption updaterOption = new MultiChoiceOption(
            "updater", 'n', "Updater to use", new String[]{
                "DirectUpdater", "OneBitUpdater"},
            new String[]{"Simple Updater without regularization ", "Simple Updater without regularization "
            }, 0);
    public MultiChoiceOption lossOption = new MultiChoiceOption(
            "loss", 'l', "Loss function", new String[]{
                "Hinge"},
            new String[]{"Hinge function"
            }, 0);
    
	public IntOption minbatchSizeOption = new IntOption( "minibatchSize", 's', "Minibatch size", 1, 1, Integer.MAX_VALUE);
	public MultiChoiceOption contextUsedOption = new MultiChoiceOption("contextUsed", 'c', "Context Type",
			new String[] { "CPU", "OPENCL", "HSA" }, new String[] { "CPU single thread",
					"OpenCL offload. Use OPENCL_DEVICE Env. variable to select device", "HSA Offload" },
			0);
	

    @Override
	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		// TODO Auto-generated method stub
		super.prepareForUseImpl(monitor, repository);
		if (contextUsedOption.getChosenIndex() == 0)
			m_context = new Context(Context.Memory.MAIN_MEMORY, null);
		else if (contextUsedOption.getChosenIndex() == 1)
			m_context = new Context(Context.Memory.OPENCL_MEMORY, null);
		else if (contextUsedOption.getChosenIndex() == 2)
			m_context = new Context(Context.Memory.HSA_MEMORY, null);
		m_hogwild_scheme = null;
		m_reference_sgd = new ReferenceSGD();
		m_reference_sgd.prepareForUseImpl(monitor, repository);
	}

	@Override
    public boolean isRandomizable() {
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (inst == null) {
            return null;
        }
        double [] votes1 = 	m_hogwild_scheme.getVotesForInstance(inst);;
     /*   double [] votes2 = m_reference_sgd.getVotesForInstance(inst);
        for (int i = 0; i < votes1.length  ; ++i)
        	if (votes1[i] != votes2[i])
        	{
        		double[] weights_hogwild = m_hogwild_scheme.getWeights();
        		double[] bias = m_hogwild_scheme.getBias();
        		DoubleVector[] weights_sgd = m_reference_sgd.getWeights();
        		for (int ww = 0; ww < weights_sgd.length ; ++ww)
        		{
        			DoubleVector vec = weights_sgd[ww];
        			double[] ref = vec.getArrayRef();
        			for (int k = 0;k < ref.length; ++k)
        			{
        				if (k == inst.classIndex())
        					continue;
        				
        				if (Math.abs(ref[k] - weights_hogwild[ ww * inst.numAttributes() + k]) > 0.00001)
        					System.out.println();
        			}
        		}
        	}*/
		return votes1;
    }


	@Override
    public void resetLearningImpl() {
		m_step = 0;
		// todo - communicate next shared memory segment. 
		// for now just reset steps 
    }
	
	private UnitOfWork m_work;
	private int m_step;
    private HogwildScheme backup;
    @Override
    public void trainOnInstanceImpl(Instance inst) {
   // 	m_reference_sgd.trainOnInstanceImpl((Instance)inst.copy());
    //	m_reference_sgd.trainOnInstance(inst);
    	if (m_hogwild_scheme == null)
    	{
        Updater upd = null;
        if (updaterOption.getChosenIndex() == 0)
          upd = new DirectUpdater(m_context, inst.dataset().numAttributes(), inst.dataset().numClasses(), inst.dataset().classIndex(), workerIndexOption.getValue(), this.workerCountOption.getValue());
        else
          upd = new OneBitUpdater(m_context, 
                  inst.dataset().numAttributes(), 
                  inst.dataset().numClasses(), 
                  inst.dataset().classIndex(), 
                  workerCountOption.getValue(), 
                  this.workerIndexOption.getValue(), 
                   this.staggerOption.getValue());
    		m_hogwild_scheme = new HogwildScheme(upd,m_context, inst.dataset(), this.workerIndexOption.getValue(),
    				this.minbatchSizeOption.getValue(), this.workerCountOption.getValue(), this.staggerOption.getValue());
    		//backup = new HogwildScheme(m_context, inst.dataset(), this.workerIndexOption.getValue(),
    		//		this.minbatchSizeOption.getValue());
    		try {
    			//testSupport.buildClassifier(inst.dataset());
				m_default_classifier.buildClassifier(inst.dataset());
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    	}
    	
	/*	m_work = new DenseInstanceBuffer(m_context, this.minbatchSizeOption.getValue(), inst.numAttributes());
		m_work.begin(Buffer.WRITE);
		m_work.append((Instance)inst.copy());
		m_work.commit();
		backup.trainStep(m_work, m_step);
		m_work = null;
*/
    	if (m_work == null)
    	{
    		createUnitOfWork(inst);
    	}
    	if (!m_work.append(inst))
    	{
    		m_work.commit();
    		m_hogwild_scheme.trainStep(m_work, m_step++);
    		createUnitOfWork(inst);
    	}
    	
    	
  //  	DoubleVector[] ref_vector = m_reference_sgd.getWeights();
  //  	double[] bias = m_reference_sgd.getBias();
  //  	double[] hog_bias = m_hogwild_scheme.getBias();
    	
    	//double[] backup_bias = backup.getBias();
    	
//    	for (int i = 0;i < bias.length ; ++i)
 //   		if (Math.abs(bias[i] - hog_bias[i]) > 0.0001)
 //   			System.out.println("aa");
    //	double[] my_weights = m_hogwild_scheme.getWeights();
  //  	System.out.println();;
    }

	private void createUnitOfWork(Instance inst) {
		if (inst instanceof SparseInstance)
			m_work = new SparseInstanceBuffer(m_context, this.minbatchSizeOption.getValue(), inst.numAttributes(), 0.5);
		else
			m_work = new DenseInstanceBuffer(m_context, this.minbatchSizeOption.getValue(), inst.numAttributes());
		m_work.begin(Buffer.WRITE);
	}

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
    	return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        out.append("Hogwild! based SGD implementation");
    }
    
  
    
}