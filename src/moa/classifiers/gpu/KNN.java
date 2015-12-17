package moa.classifiers.gpu;



import org.moa.gpu.SlidingWindow;
import org.moa.opencl.knn.Search;
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
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Naive implementation backed by the native library.
 *
 * @author bsp
 *
 */
public class KNN extends AbstractClassifier  {

	
	private int m_batch_size;
    private boolean m_native_init = false;
    private transient SlidingWindow m_sliding_window;
    
    private int m_k;
    private int m_distance_weighting;
    private ZeroR m_default_classifier = new ZeroR();
    private transient Context m_context;
    private transient Search m_search;
    
    public KNN()
    {
    }

    public ClassOption searchMethod = new ClassOption("searchMethod", 'm', "KNN search Method", Search.class, "org.moa.opencl.knn.SimpleZOrderSearch");
    		
    public IntOption neighboursNumber = new IntOption("neighbourNumber", 'n', "Number of neighbours to use", 16, 1, Integer.MAX_VALUE);
    public IntOption slidingWindowSize = new IntOption("slidingWindowSize", 'b', "Sliding Window Size", 32768, 2, Integer.MAX_VALUE);
    public MultiChoiceOption distanceWeightingOption = new MultiChoiceOption("distanceWeighting", 'w', "Distance Weighting", 
            new String[]{"none", "similarity", "inverse"}, 
            new String[]{"No distance weighting", "Weight by 1-distance", "Weight by 1/distance"}, 0);

   
    

    @Override
	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		// TODO Auto-generated method stub
		super.prepareForUseImpl(monitor, repository);
		m_context = GlobalContext.context();
		m_search = (Search) searchMethod.materializeObject(monitor, repository);
		
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
        if (m_sliding_window == null || !m_sliding_window.isReady())
        {
			try {
				return m_default_classifier.distributionForInstance(inst);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return null;
        }
        m_sliding_window.commit(); // commit all data model changes to gpu
        
        double[] res = null;
		try {
			res = m_search.getVotesForInstance(inst, m_sliding_window.model(), m_k);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        return res;
    }


	@Override
    public void resetLearningImpl() {
        m_batch_size = slidingWindowSize.getValue();
        m_k = neighboursNumber.getValue();
        m_distance_weighting = distanceWeightingOption.getChosenIndex();
        shutdown();
    }
    
    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	if (m_sliding_window == null)
    	{
    		m_sliding_window = new SlidingWindow(m_context, inst.dataset(),m_batch_size);
    		
    		m_search.init(m_context, inst.dataset());
    		m_search.setSlidingWindow(m_sliding_window);
    		
    		try {
				m_default_classifier.buildClassifier(inst.dataset());
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    	}
    	m_sliding_window.begin(); // make sure buffer is mapped
        m_sliding_window.update(inst); // update buffer
        m_search.markDirty();
        
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        out.append("Naive KNN implementation backed by viennacl ");
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
