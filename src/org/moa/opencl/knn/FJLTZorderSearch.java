package org.moa.opencl.knn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.FJLT;
import org.moa.gpu.SlidingWindow;
import org.moa.opencl.util.BufHelper;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.moa.opencl.util.Distance;
import org.moa.opencl.util.DoubleMergeSort;
import org.moa.opencl.util.MinMax;
import org.moa.opencl.util.NarySearch;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import moa.classifiers.gpu.zorder.ProjectedZOrderTransform;
import moa.classifiers.gpu.zorder.ZOrderItem;
import moa.classifiers.gpu.zorder.ZOrderSequence;
import moa.classifiers.gpu.zorder.ZOrderTransform;
import moa.classifiers.lazy.neighboursearch.EuclideanDistance;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.tasks.TaskMonitor;
import org.moa.opencl.util.Operations;
import weka.core.Instance;
import weka.core.Instances;

//WEKA
//valuation instances,total train time,total train speed,last train time,last train speed,test time,test speed,classified instances,classifications correct (percent),Kappa Statistic (percent),Kappa Temporal Statistic (percent),model training instances,model serialized size (bytes)
//35000.0,6.96875,5022.421524663677,6.96875,5022.421524663677,395.609375,252.7745961530866,100000.0,95.705,91.41000477603734,91.39933517561775,35000.0,0.0

//evaluation instances,total train time,total train speed,last train time,last train speed,test time,test speed,classified instances,classifications correct (percent),Kappa Statistic (percent),Kappa Temporal Statistic (percent),model training instances,model serialized size (bytes)
//10000.0,0.484375,20645.16129032258,0.484375,20645.16129032258,71.140625,1405.6665934548648,100000.0,42.321,0.0,-18.49089937959654,10000.0,0.0

/* 
 * This algorithm uses a number of z-order curves to generate - 21 sec baseline  
 */
public class FJLTZorderSearch extends Search{
	private boolean m_compute_approximation_error =false;
	private int total_instances = 0;
	private double ratio = 0;
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
	private ProjectedZOrderTransform[] m_transform;
	private Buffer m_result_index_buffer;
	private Buffer m_candidates_buffer;
	private ArrayList<ZOrderSequence> m_z_orders;
	private CLogsVarKeyJava m_sort;
	private Operations m_ops;
	private int m_number_of_curves;
	private ArrayList<Buffer> m_random_shift_vectors;
	private ArrayList<Buffer> m_z_order_sequences;
	private EuclideanDistance m_cpu_distance;
	private NarySearch m_search;
	private int m_projected_dims;
	private CLogsVarKeyJava m_curve_sort;
	

	public FJLTZorderSearch(int num_curves, int num_dims)
	{
		m_dirty = true;
		m_number_of_curves =num_curves;
		m_projected_dims = num_dims;
	}
	
	public void init(Context ctx, Instances dataset)
	{
		Random rnd = new Random(System.currentTimeMillis());
		m_dataset = dataset; 
		m_context = ctx;
		m_distance = new Distance(ctx);
		
		m_sort = new CLogsVarKeyJava(m_context, true);
		m_curve_sort = new CLogsVarKeyJava(m_context, false);
		m_ops = new Operations(m_context);
		m_min_max = new MinMax(ctx);
		m_search = new NarySearch(ctx,true); 
	
		m_min_values = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.FLOAT_SIZE);
		m_max_values = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.FLOAT_SIZE);
		m_test_instance =  new DenseInstanceBuffer(DenseInstanceBuffer.Kind.FLOAT_BUFFER, ctx, 1, m_dataset.numAttributes());
		m_min_values_with_test_instance = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.FLOAT_SIZE);
		m_max_values_with_test_instance = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.FLOAT_SIZE);
		
		m_attribute_types = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.INT_SIZE);
		m_attribute_types.mapBuffer(Buffer.WRITE);
		m_attribute_types.writeArray(0, attributeTypes(dataset));
		m_attribute_types.commitBuffer();
		
		m_random_shift_vectors = new ArrayList<Buffer>();
		for (int i = 0; i < m_number_of_curves; ++i)
		{
			Buffer rnd_buffer = new Buffer(m_context, m_dataset.numAttributes()* DirectMemory.FLOAT_SIZE);
			float[] random_vector = new float[dataset.numAttributes()];
			for (int j = 0; j < random_vector.length ; ++j)
			{
				random_vector[j] = rnd.nextFloat()/1000;
			}
			rnd_buffer.mapBuffer(Buffer.WRITE);
			rnd_buffer.writeArray(0,  random_vector);
			rnd_buffer.commitBuffer();
			m_random_shift_vectors.add(rnd_buffer);
		}
				
		m_cpu_distance = new EuclideanDistance(dataset);
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
			//System.out.println("reinit variables");
			m_min_max.fullMinMaxFloat(instance.dataset(), data, m_min_values, m_max_values);
			m_result_buffer = new Buffer(m_context, data.rows() * DirectMemory.FLOAT_SIZE);
			m_result_index_buffer = new Buffer(m_context, data.rows() * DirectMemory.INT_SIZE);
			m_transform = new ProjectedZOrderTransform[m_number_of_curves];
			for (int i = 0; i < m_number_of_curves; ++i)
				m_transform[i] = new ProjectedZOrderTransform(m_context, m_curve_sort,instance.dataset().numAttributes(), data.rows(), m_projected_dims);
		}
		
		if (m_z_orders == null)
		{
			for (int i = 0; i < m_number_of_curves; ++i)
			{
				m_transform[i].fillNormalizedData(instance.dataset(), data);
				m_transform[i].createDeviceRandomShiftZOrder(m_random_shift_vectors.get(i));
			}
			m_dirty = false;
			m_z_orders = new ArrayList<ZOrderSequence>();
			m_context.finishDefaultQueue();
		}
		

		
    //System.out.println("Evaluating");
		
		m_test_instance.begin(Buffer.WRITE);
		m_test_instance.set(instance, 0);
		m_test_instance.commit();
		
		
		m_min_values.copyTo(m_min_values_with_test_instance);
		m_max_values.copyTo(m_max_values_with_test_instance);
		m_min_max.updateMinMaxFloat(m_dataset, m_test_instance, m_min_values_with_test_instance, m_max_values_with_test_instance);
		HashSet<Integer> possible_candidates = new HashSet<Integer>();
		int index = 0;
		int strange = 0;
		
		for (int curve = 0; curve < m_transform.length ; ++curve)
			m_transform[curve].enqueueSearch(  
					m_random_shift_vectors.get(index++), 
					m_test_instance.attributes());
		
		for (int curve = 0; curve < m_transform.length ; ++curve)
		{
      //System.out.println("Next item");
  
			int [] candidates = m_transform[curve].canidatesForInstance(K);
			if (candidates.length < K)
				System.out.println("error");
			
	
			
			for (int next: candidates)
				possible_candidates.add(next);
 
		}
		

    //System.out.println("Flush candidates");
		int[] cnd_items = new int[possible_candidates.size()];
		index = 0;
		for (Integer i : possible_candidates)
			cnd_items[index++] = i;
		

		if (m_candidates_buffer == null) 
			m_candidates_buffer = new Buffer(m_context, data.rows() * DirectMemory.INT_SIZE);
		
		m_candidates_buffer.mapBuffer(Buffer.WRITE);
		m_candidates_buffer.writeArray(0, cnd_items);
		m_candidates_buffer.commitBuffer();
		int sort_size = cnd_items.length;
		m_distance.squareDistanceFloat(
				m_dataset, 
				m_test_instance, 
				data, 
				m_min_values_with_test_instance, 
				m_max_values_with_test_instance, 
				m_attribute_types, 
				m_result_buffer, 
				cnd_items.length, 
				m_candidates_buffer);
    	
		m_ops.prepareOrderKey(m_result_index_buffer, sort_size);
		m_sort.sortFixedBuffer(m_result_buffer, m_result_index_buffer, sort_size);
		int[] nearest_k = new int[K];
		m_result_index_buffer.mapBuffer(Buffer.READ, 0, K * DirectMemory.INT_SIZE);
		m_result_index_buffer.readArray(0, nearest_k);
		m_result_index_buffer.commitBuffer();
		//float[] found_dist = BufHelper.rbf(m_result_buffer)
		
		
		if (false)
		{
      //System.out.println("Compute error");
			m_result_buffer.mapBuffer(Buffer.READ);
			double kth_approx = m_result_buffer.read(1 * DirectMemory.FLOAT_SIZE);
			m_result_buffer.commitBuffer();
			
			m_distance.squareDistanceFloat(m_dataset, 
					m_test_instance, 
					data, 
					m_min_values_with_test_instance, 
					m_max_values_with_test_instance, 
					m_attribute_types, 
					m_result_buffer);
			
			m_ops.prepareOrderKey(m_result_index_buffer, data.rows());
			m_sort.sortFixedBuffer(m_result_buffer, m_result_index_buffer, data.rows());
			double kth_true;
			m_result_buffer.mapBuffer(Buffer.READ);
			kth_true = m_result_buffer.read(1 * DirectMemory.FLOAT_SIZE);
			
			m_result_buffer.commitBuffer();
			
			double eps = kth_approx/kth_true;
			if (eps > 1000)
			{
				float[] atrrs = BufHelper.rbf(m_test_instance.attributes());
				float[] distances = BufHelper.rbf(m_result_buffer);
				float[] mins = BufHelper.rbf(m_min_values_with_test_instance);
				float[] maxs = BufHelper.rbf(m_max_values_with_test_instance);
				System.out.println();;
			}
			
			
			ratio += eps;
			total_instances++;
			System.out.println("Current eps = " + eps + " avg "+ ratio/total_instances);
		
		}
		//System.out.println("Done");
		if (cnd_items.length == 1)
			// a bug but ignore it for now
			return new double[K];
		return makeDistribution(nearest_k, cnd_items, m_result_buffer, K);
    
	}
	
	public void resetRelativeError()
	{
		ratio = 0;
		total_instances = 0;
	}
	
	public double getRelativeError() 
	{
		return 1;//ratio/total_instances;
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
