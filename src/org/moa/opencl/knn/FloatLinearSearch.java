package org.moa.opencl.knn;

import org.moa.gpu.DenseInstanceBuffer;


import org.moa.gpu.SlidingWindow;
import org.moa.opencl.util.Distance;
import org.moa.opencl.util.DoubleMergeSort;
import org.moa.opencl.util.MinMax;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.tasks.TaskMonitor;
import weka.core.Instance;
import weka.core.Instances;

// 32K
//evaluation instances,total train time,total train speed,last train time,last train speed,test time,test speed,classified instances,classifications correct (percent),Kappa Statistic (percent),Kappa Temporal Statistic (percent),model training instances,model serialized size (bytes)
//35000.0,0.40625,86153.84615384616,0.40625,86153.84615384616,50.359375,1985.7275829972075,100000.0,95.499,90.99807707806484,90.98682366134007,35000.0,0.0
//70000.0,0.4375,160000.0,0.03125,1120000.0,49.796875,2008.1581424537183,100000.0,95.422,90.84369109910038,90.87265984807702,70000.0,0.0

//EvaluatePeriodicHeldOutTest -l gpu.KNN
//valuation instances,total train time,total train speed,last train time,last train speed,test time,test speed,classified instances,classifications correct (percent),Kappa Statistic (percent),Kappa Temporal Statistic (percent),model training instances,model serialized size (bytes)
//100000.0,0.609375,164102.5641025641,0.609375,164102.5641025641,386.140625,2589.7301015659773,1000000.0,76.4087,51.912289570814664,51.6706409689103,100000.0,0.0
public class FloatLinearSearch extends Search{
	
	private Instances m_dataset;
	private Context m_context;
	private MinMax m_min_max;
	private Buffer m_min_values;
	private Buffer m_max_values;
	private Buffer m_min_values_with_test_instance;
	private Buffer m_max_values_with_test_instance;
	private DenseInstanceBuffer m_test_instance;
	private Buffer m_result_buffer;
	private Buffer m_attribute_types;
	private Distance m_distance;
	private DoubleMergeSort m_sort;
	private Buffer m_result_index_buffer;

	public FloatLinearSearch()
	{
		m_dirty = true;
	}
	
	public void init(Context ctx, Instances dataset)
	{
		m_dataset = dataset; 
		m_context = ctx;
		m_distance = new Distance(ctx);
		m_min_max = new MinMax(ctx);
		
		m_min_values = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.DOUBLE_SIZE);
		m_max_values = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.DOUBLE_SIZE);
		m_test_instance =  new DenseInstanceBuffer(ctx, 1, m_dataset.numAttributes());
		m_min_values_with_test_instance = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.DOUBLE_SIZE);
		m_max_values_with_test_instance = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.DOUBLE_SIZE);
		
		m_attribute_types = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.INT_SIZE);
		m_attribute_types.mapBuffer(Buffer.WRITE);
		m_attribute_types.writeArray(0, attributeTypes(dataset));
		m_attribute_types.commitBuffer();
		
		
		
	}
	
    private int[] attributeTypes(Instances dataset)
    {
        int[] attributeTypes = new int[dataset.numAttributes()];
         for (int i = 0 ;i < attributeTypes.length; ++i)
         {
             if(dataset.attribute(i).isNumeric())
                 attributeTypes[i] = 0;
             else
             if(dataset.attribute(i).isNominal())
                 attributeTypes[i] = 1;
             else
                 attributeTypes[i] = 2;
         }
        return attributeTypes;
    }

	
	public double[] getVotesForInstance(Instance instance, DenseInstanceBuffer data, int K ) throws Exception
	{
		
		if (m_result_buffer == null)
		{
			m_result_buffer = new Buffer(m_context, data.rows() * DirectMemory.DOUBLE_SIZE);
			m_result_index_buffer = new Buffer(m_context, data.rows() * DirectMemory.INT_SIZE);
			m_sort = new DoubleMergeSort(m_context, data.rows());
		}

		if (m_dirty)
		{
			m_min_max.fullMinMaxDouble(m_dataset, data, m_min_values, m_max_values);
			m_dirty = false;
		}
		m_min_values.copyTo(m_min_values_with_test_instance);
		m_max_values.copyTo(m_max_values_with_test_instance);
		m_test_instance.begin(Buffer.WRITE);
		m_test_instance.set(instance, 0);
		m_test_instance.commit();
		m_min_max.updateMinMaxDouble(m_dataset, m_test_instance, m_min_values_with_test_instance, m_max_values_with_test_instance);
		
		m_distance.squareDistance(m_dataset, 
				m_test_instance, 
				data, 
				m_min_values, 
				m_max_values, 
				m_attribute_types, 
				m_result_buffer);
		
		m_sort.sort(m_result_buffer, m_result_index_buffer);
		int[] candidates = new int[K];
		m_result_index_buffer.mapBuffer(Buffer.READ, 0, K * DirectMemory.INT_SIZE);
		m_result_index_buffer.readArray(0, candidates);
		m_result_index_buffer.commitBuffer();
		
		return makeDistribution(candidates, m_result_buffer, K);
	}
	
	

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		// TODO Auto-generated method stub
		
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		// TODO Auto-generated method stub
		
	}

}
