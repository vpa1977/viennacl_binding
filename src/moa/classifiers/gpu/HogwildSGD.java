package moa.classifiers.gpu;



import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadPoolExecutor;

import org.moa.gpu.SlidingWindow;
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
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import moa.tasks.TaskMonitor;
import test.org.moa.opencl.IBk;
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;

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
    
	
    private ZeroR m_default_classifier = new ZeroR();
    private transient Context m_context;
    private transient HogwildScheme m_hogwild_scheme;
    
    public HogwildSGD()
    {
    }
    
    public MultiChoiceOption updaterOption = new MultiChoiceOption(
            "updater", 'n', "Updater to use", new String[]{
                "SimpleUpdater"},
            new String[]{"Simple Updater without regularization "
            }, 0);
    public MultiChoiceOption lossOption = new MultiChoiceOption(
            "loss", 'l', "Loss function", new String[]{
                "Hinge"},
            new String[]{"Hinge function"
            }, 0);
    
	public IntOption parallelBatchesOption = new IntOption( "parallelBatches", 'p', "Number of parallel batches", 16, 1, Integer.MAX_VALUE);
	public IntOption minbatchSizeOption = new IntOption( "minibatchSize", 's', "Minibatch size", 16, 1, Integer.MAX_VALUE);
    public IntOption slidingWindowSizeOption = new IntOption("slidingWindowSize", 'b', "Number of minibatches in Sliding Window Size", 10, 2, Integer.MAX_VALUE);
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
		return 	m_hogwild_scheme.getVotesForInstance(inst);
    }


	@Override
    public void resetLearningImpl() {
		
    }
	
	private UnitOfWork m_work;
    
    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	if (m_hogwild_scheme == null)
    	{
    		m_hogwild_scheme = new HogwildScheme(m_context, inst.dataset(), this.parallelBatchesOption.getValue(),
    				this.minbatchSizeOption.getValue());
    		m_hogwild_scheme.populate(true);
    		try {
    			//testSupport.buildClassifier(inst.dataset());
				m_default_classifier.buildClassifier(inst.dataset());
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    	}
    	if (m_work == null)
    	{
    		m_work = m_hogwild_scheme.take();
    	}
    	if (!m_work.append(inst))
    	{
    		m_work.commit();
    		m_hogwild_scheme.put(m_work);
    		m_work = m_hogwild_scheme.take();
    		m_work.append(inst);
    	}
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
