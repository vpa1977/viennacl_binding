package moa.classifiers.gpu.kdtree;

import org.moa.gpu.DenseInstanceBuffer;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import moa.classifiers.lazy.kNN;
import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.KdTreeParallelDistance;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;
import moa.options.FlagOption;
import weka.core.Instance;
import weka.core.Instances;

public class KNNExtended extends kNN{
	
	static
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
	private int C=0;
	
	private DenseInstanceBuffer m_buffer;
	private Buffer m_indices;

	private Context m_context;
	
	public FlagOption useGPUOption = new FlagOption("useGPU", 'g', "Run search on GPU or CPU");

		
	 @Override
	public void trainOnInstanceImpl(Instance inst) {
		if (inst.classValue() > C)
			C = (int)inst.classValue();
		// TODO Auto-generated method stub
		this.search = null;
		if (m_buffer == null) 
		{
			m_context = new Context(Context.DEFAULT_MEMORY, null);
			m_buffer = new DenseInstanceBuffer(m_context, this.limitOption.getValue(), inst.numAttributes());
			m_indices = new Buffer(m_context, DirectMemory.INT_SIZE * this.limitOption.getValue());
		}
		super.trainOnInstanceImpl(inst);
	}
	 
	    @Override
    public void resetLearningImpl() {
		this.window = null;
		this.search = null;
    }

  
  private NearestNeighbourSearch search;

	public double[] getVotesForInstance(Instance inst) {
			double v[] = new double[C+1];
			try {
				if (search == null)
				{
					if (this.nearestNeighbourSearchOption.getChosenIndex()== 0) {
						search = new LinearNNSearch(this.window);  
					} else {
						if (useGPUOption.isSet())
							search = new KdTreeParallelDistance(m_context,m_buffer, m_indices);
						else
							search = new KDTree();
					    search.setInstances(this.window);
					}
				}
				
				if (this.window.numInstances()>0) {	
					Instances neighbours = search.kNearestNeighbours(inst,Math.min(kOption.getValue(),this.window.numInstances()));
					for(int i = 0; i < neighbours.numInstances(); i++) {
						v[(int)neighbours.instance(i).classValue()]++;
					}
				}
			} catch(Exception e) {
				//System.err.println("Error: kNN search failed.");
				e.printStackTrace();
				//System.exit(1);
				return new double[inst.numClasses()];
			}
			return v;
	    }
}
