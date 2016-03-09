package moa.classifiers.gpu;



import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.SlidingWindow;
import org.moa.opencl.knn.DeviceZOrderSearch;
import org.moa.opencl.knn.DoubleCosineSearch;
import org.moa.opencl.knn.DoubleLinearSearch;
import org.moa.opencl.knn.FJLTZorderSearch;
import org.moa.opencl.knn.FloatLinearSearch;
import org.moa.opencl.knn.Search;
import org.moa.opencl.knn.SimpleZOrderSearch;
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
public class KNNZOrder extends AbstractClassifier  {

	
	private int m_batch_size;
    private boolean m_native_init = false;
    private transient SlidingWindow m_sliding_window;
    
    private int m_k;
    private int m_distance_weighting;
    private ZeroR m_default_classifier = new ZeroR();
    private transient Context m_context;
    private transient Search m_search;
    
    private IBk testSupport = null;
    
    static
    {
    	System.loadLibrary("viennacl-java-binding");
    }
    
    
    public KNNZOrder()
    {
    }
    
    public MultiChoiceOption nearestNeighbourSearchOption = new MultiChoiceOption(
            "nearestNeighbourSearch", 'n', "Nearest Neighbour Search to use", new String[]{
                "Z-OrderShift (device curves)"},
            new String[]{
                "Z-Order shift search algorithm for nearest neighbour search"
            }, 0);
    		
    public IntOption nCurvesOption = new IntOption( "nCurves", 't', "The number of curves", 10, 1, Integer.MAX_VALUE);
    public IntOption nTargetDimOption = new IntOption( "nTargetDim", 'd', "Projected dimensions", 4, 1, Integer.MAX_VALUE);
	public IntOption kOption = new IntOption( "k", 'k', "The number of neighbors", 10, 1, Integer.MAX_VALUE);
    public IntOption slidingWindowSizeOption = new IntOption("slidingWindowSize", 'b', "Sliding Window Size", 32768, 2, Integer.MAX_VALUE);
    public MultiChoiceOption distanceWeightingOption = new MultiChoiceOption("distanceWeighting", 'w', "Distance Weighting", 
            new String[]{"none", "similarity", "inverse"}, 
            new String[]{"No distance weighting", "Weight by 1-distance", "Weight by 1/distance"}, 0);

   
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
      
		if (nearestNeighbourSearchOption.getChosenIndex() == 0)
			m_search = new FJLTZorderSearch(nCurvesOption.getValue(), nTargetDimOption.getValue());
		
	//	testSupport = new IBk(m_k);
		///testSupport.setWindowSize(1000); 
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
		//	double[] supp = testSupport.distributionForInstance(inst);
		//	for (int i = 0; i < supp.length; ++i)
		//	{
		//		if (res[i] != supp[i])
		//			System.out.println("Break");
		//	}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        return res;
    }


	@Override
    public void resetLearningImpl() {
        m_batch_size = slidingWindowSizeOption.getValue();
        m_k = kOption.getValue();
        m_distance_weighting = distanceWeightingOption.getChosenIndex();
        m_sliding_window = null;
        shutdown();
    }
    
    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	if (m_sliding_window == null)
    	{
    		m_sliding_window = new SlidingWindow(DenseInstanceBuffer.Kind.FLOAT_BUFFER, m_context, inst.dataset(),m_batch_size);
    		
    		m_search.init(m_context, inst.dataset());
    		m_search.setSlidingWindow(m_sliding_window);
    		
    		
    		try {
    			//testSupport.buildClassifier(inst.dataset());
				m_default_classifier.buildClassifier(inst.dataset());
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    	}
    	
    	/*try {
			testSupport.updateClassifier(inst);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/
    	( (FJLTZorderSearch) m_search).resetRelativeError();
    	m_sliding_window.begin(); // make sure buffer is mapped
        m_sliding_window.update(inst); // update buffer
        m_search.markDirty();
        m_search.update(inst);
        
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{
        		new Measurement("eps relative error",  ((FJLTZorderSearch)m_search).getRelativeError())
        };
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
