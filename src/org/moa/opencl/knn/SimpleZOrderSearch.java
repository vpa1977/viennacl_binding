package org.moa.opencl.knn;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.SlidingWindow;
import org.moa.opencl.util.Distance;
import org.moa.opencl.util.DoubleMergeSort;
import org.moa.opencl.util.MinMax;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import moa.classifiers.gpu.zorder.ZOrderItem;
import moa.classifiers.gpu.zorder.ZOrderTransform;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.tasks.TaskMonitor;
import weka.core.Instance;
import weka.core.Instances;

//WEKA
//valuation instances,total train time,total train speed,last train time,last train speed,test time,test speed,classified instances,classifications correct (percent),Kappa Statistic (percent),Kappa Temporal Statistic (percent),model training instances,model serialized size (bytes)
//35000.0,6.96875,5022.421524663677,6.96875,5022.421524663677,395.609375,252.7745961530866,100000.0,95.705,91.41000477603734,91.39933517561775,35000.0,0.0

//evaluation instances,total train time,total train speed,last train time,last train speed,test time,test speed,classified instances,classifications correct (percent),Kappa Statistic (percent),Kappa Temporal Statistic (percent),model training instances,model serialized size (bytes)
//10000.0,0.484375,20645.16129032258,0.484375,20645.16129032258,71.140625,1405.6665934548648,100000.0,42.321,0.0,-18.49089937959654,10000.0,0.0
public class SimpleZOrderSearch extends Search{
	
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
	private ZOrderTransform m_transform;
	private Buffer m_result_index_buffer;
	private Buffer m_candidates_buffer;
	private ZOrderItem[] m_z_orders;
	private DoubleMergeSort m_sort;

	public SimpleZOrderSearch()
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
			m_transform = new ZOrderTransform(m_context, instance.dataset().numAttributes(), data.rows());
			m_candidates_buffer = new Buffer(m_context, (2*K+1) * DirectMemory.INT_SIZE);
			m_sort = new DoubleMergeSort(m_context, data.rows());
		}

		if (m_z_orders == null)
		{
			m_min_max.fullMinMaxDouble(m_dataset, data, m_min_values, m_max_values);
			m_z_orders = m_transform.createZOrder(instance.dataset(), data, m_min_values, m_max_values, m_attribute_types, true);
			m_dirty = false;
		}
		m_min_values.copyTo(m_min_values_with_test_instance);
		m_max_values.copyTo(m_max_values_with_test_instance);
		m_test_instance.begin(Buffer.WRITE);
		m_test_instance.set(instance, 0);
		m_test_instance.commit();
		m_min_max.updateMinMaxDouble(m_dataset, m_test_instance, m_min_values_with_test_instance, m_max_values_with_test_instance);
		
		ZOrderItem[] possible_k = m_transform.findInZOrder(instance.dataset(), m_z_orders, m_test_instance.attributes(), m_min_values, m_max_values, m_attribute_types, K);
		int[] cnd_items = new int[possible_k.length];
		for (int i = 0; i < possible_k.length ; ++i)
			cnd_items[i] = possible_k[i].instanceIndex();
		m_candidates_buffer.mapBuffer(Buffer.WRITE);
		m_candidates_buffer.writeArray(0, cnd_items);
		m_candidates_buffer.commitBuffer();
		
		m_distance.squareDistance(
				possible_k.length,
				m_dataset, 
				m_test_instance, 
				data, 
				m_min_values, 
				m_max_values, 
				m_attribute_types, 
				m_result_buffer, 
				m_candidates_buffer);
		
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
