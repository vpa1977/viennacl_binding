package org.moa.opencl.knn;

import org.moa.gpu.DenseInstanceBuffer;


import org.moa.gpu.SlidingWindow;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.moa.opencl.util.Distance;
import org.moa.opencl.util.DoubleMergeSort;
import org.moa.opencl.util.MinMax;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.tasks.TaskMonitor;
import test.org.moa.opencl.util.CLogsJavaTest;

import org.moa.opencl.util.Operations;
import weka.core.Instance;
import weka.core.Instances;
//WEKA
//valuation instances,total train time,total train speed,last train time,last train speed,test time,test speed,classified instances,classifications correct (percent),Kappa Statistic (percent),Kappa Temporal Statistic (percent),model training instances,model serialized size (bytes)
//35000.0,6.96875,5022.421524663677,6.96875,5022.421524663677,395.609375,252.7745961530866,100000.0,95.705,91.41000477603734,91.39933517561775,35000.0,0.0

//EvaluatePeriodicHeldOutTest -l gpu.KNN
//valuation instances,total train time,total train speed,last train time,last train speed,test time,test speed,classified instances,classifications correct (percent),Kappa Statistic (percent),Kappa Temporal Statistic (percent),model training instances,model serialized size (bytes)
//100000.0,0.609375,164102.5641025641,0.609375,164102.5641025641,386.140625,2589.7301015659773,1000000.0,76.4087,51.912289570814664,51.6706409689103,100000.0,0.0
public class DoubleLinearSearch extends Search{
	
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
	//private DoubleMergeSort m_sort;
	private CLogsVarKeyJava m_sort;
  private Operations m_ops;
	private Buffer m_result_index_buffer;

	public DoubleLinearSearch()
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
		//	m_sort = new DoubleMergeSort(m_context, data.rows());
			m_sort = new CLogsVarKeyJava(m_context, true, "unsigned long", "unsigned int");
			m_ops = new Operations(m_context);
		
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
		//m_min_max.updateMinMaxDouble(m_dataset, m_test_instance, m_min_values, m_max_values);
		
		m_distance.squareDistance(m_dataset, 
				m_test_instance, 
				data, 
				m_min_values, 
				m_max_values, 
				m_attribute_types, 
				m_result_buffer);
		
/*		double[] distances = new double[(int)(m_result_buffer.byteSize()/DirectMemory.DOUBLE_SIZE)];
		m_result_buffer.mapBuffer(Buffer.READ);
		m_result_buffer.readArray(0, distances);
		m_result_buffer.commitBuffer();

		System.out.print("distances opencl ");
	    for (int i = 0; i < distances.length; ++i)
	    	System.out.print(" "+ Math.sqrt(distances[i]));
	    System.out.println();
*/
		int size = (int)(m_result_buffer.byteSize()/DirectMemory.DOUBLE_SIZE);
		m_ops.prepareOrderKey(m_result_index_buffer, size);
		m_sort.sortFixedBuffer(m_result_buffer, m_result_index_buffer,size);
		int[] candidates = new int[K];
		m_result_index_buffer.mapBuffer(Buffer.READ, 0, (K) * DirectMemory.INT_SIZE);
		m_result_index_buffer.readArray(0, candidates);
		m_result_index_buffer.commitBuffer();

	//	m_sort.getDstIndex().mapBuffer(Buffer.READ);
	//	m_sort.getDstIndex().readArray(0, candidates);
	//	m_sort.getDstIndex().commitBuffer();
		
		
	/*	data.begin(Buffer.READ);   
		System.out.println("Neigh");
		for (int i = 0; i < candidates.length; ++i)
		{
			System.out.println(data.read(candidates[i], m_dataset));
		}
		data.commit();
				*/
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
