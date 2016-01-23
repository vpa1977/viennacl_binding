package org.moa.opencl.knn.tree;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.opencl.knn.Search;
import org.viennacl.binding.Context;

import moa.core.ObjectRepository;
import moa.tasks.TaskMonitor;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.KDTree;

public class TreeSearch extends Search {

	private int m_tree_depth = 10;
	
	private Context m_context;
	private Instances m_dataset;
	private KDTreeBufferCPU m_buffer; 
	
	@Override
	public void getDescription(StringBuilder sb, int indent) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void init(Context context, Instances dataset) {
		m_context = context;
		m_dataset = dataset;
	}

	@Override
	public double[] getVotesForInstance(Instance instance, DenseInstanceBuffer data, int K) throws Exception {
		if (m_dirty)
		{
			buildTree();
			m_dirty = false;
		}
		return null;
	}

	private void buildTree() {
		//m_buffer = new KDTreeBuffer(m_tree_depth);
		
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		// TODO Auto-generated method stub
		
	}

}
