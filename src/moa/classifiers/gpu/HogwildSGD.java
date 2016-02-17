package moa.classifiers.gpu;



import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadPoolExecutor;

import org.moa.gpu.SlidingWindow;
import org.moa.gpu.SparseSlidingWindow;
import org.moa.opencl.knn.DeviceZOrderSearch;
import org.moa.opencl.knn.DoubleLinearSearch;
import org.moa.opencl.knn.Search;
import org.moa.opencl.knn.SimpleZOrderSearch;
import org.moa.opencl.sgd.Gradient;
import org.moa.opencl.sgd.HingeGradient;
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

	
	private int m_batch_size;
    private boolean m_native_init = false;
    private transient SparseSlidingWindow m_sliding_window;
    private ZeroR m_default_classifier = new ZeroR();
    private transient Context m_context;
    
    
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
    
	public IntOption minbatchSizeOption = new IntOption( "minibatchSize", 's', "Minibatch size", 128, 1, Integer.MAX_VALUE);
    public IntOption slidingWindowSizeOption = new IntOption("slidingWindowSize", 'b', "Number of minibatches in Sliding Window Size", 10, 2, Integer.MAX_VALUE);
	private Updater m_updater;
	private HingeGradient m_loss_function;


	private class UpdateProcess 
	{
		private Gradient m_gradient;
		private Updater m_updater;
	}
	
	/* updaters */
	private ArrayBlockingQueue<UpdateProcess> m_update_workers = new ArrayBlockingQueue<UpdateProcess>(10);
	/** executor for training */
	private ForkJoinPool m_executor = new ForkJoinPool();
    

    @Override
	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		// TODO Auto-generated method stub
		super.prepareForUseImpl(monitor, repository);
		m_context = GlobalContext.context();
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
		try {
			return m_default_classifier.distributionForInstance(inst);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
    }


	@Override
    public void resetLearningImpl() {
		m_sliding_window = null;
        shutdown();
    }
    
    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	if (m_sliding_window == null)
    	{
    		if (updaterOption.getChosenIndex() == 0)
    			m_updater = new SimpleUpdater(m_context, inst.dataset().numAttributes());
    		
    		m_sliding_window = new SparseSlidingWindow(m_context, inst.dataset(), 
    				slidingWindowSizeOption.getValue(), 
    				minbatchSizeOption.getValue());
    		if (lossOption.getChosenIndex() == 0)
    			m_loss_function = new HingeGradient(m_context,inst.dataset().numAttributes() , minbatchSizeOption.getValue());
    		
    		
    		
    		try {
    			//testSupport.buildClassifier(inst.dataset());
				m_default_classifier.buildClassifier(inst.dataset());
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    	}
    	
        
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        out.append("Hogwild! based SGD implementation");
    }
    
    private void shutdown()
    {
        if (m_native_init)
        {
            m_sliding_window.dispose();
            m_native_init = false;
        }
    }
    
}
