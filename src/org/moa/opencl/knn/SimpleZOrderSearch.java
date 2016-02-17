package org.moa.opencl.knn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.SlidingWindow;
import org.moa.opencl.util.Distance;
import org.moa.opencl.util.DoubleMergeSort;
import org.moa.opencl.util.MinMax;
import org.moa.opencl.util.NarySearch;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import moa.classifiers.gpu.zorder.ZOrderItem;
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
 * This algorithm uses a number of z-order curves to generate  
 */
public class SimpleZOrderSearch extends Search{
	
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
	private ZOrderTransform m_transform;
	private Buffer m_result_index_buffer;
	private Buffer m_candidates_buffer;
	private ArrayList<ZOrderItem[]> m_z_orders;
	private DoubleMergeSort m_sort;
	private Operations m_ops;
	private int m_number_of_curves;
	private ArrayList<Buffer> m_random_shift_vectors;
	private EuclideanDistance m_cpu_distance;
	private NarySearch m_search;

	public SimpleZOrderSearch()
	{
		m_dirty = true;
		m_number_of_curves = 1;
	}
	
	public void init(Context ctx, Instances dataset)
	{
		Random rnd = new Random(System.currentTimeMillis());
		m_dataset = dataset; 
		m_context = ctx;
		m_distance = new Distance(ctx);
		m_min_max = new MinMax(ctx);
		m_search = new NarySearch(ctx,false);
		
		m_min_values = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.DOUBLE_SIZE);
		m_max_values = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.DOUBLE_SIZE);
		m_test_instance =  new DenseInstanceBuffer(ctx, 1, m_dataset.numAttributes());
		m_min_values_with_test_instance = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.DOUBLE_SIZE);
		m_max_values_with_test_instance = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.DOUBLE_SIZE);
		
		m_attribute_types = new Buffer(ctx, m_dataset.numAttributes() * DirectMemory.INT_SIZE);
		m_attribute_types.mapBuffer(Buffer.WRITE);
		m_attribute_types.writeArray(0, attributeTypes(dataset));
		m_attribute_types.commitBuffer();
		
		m_random_shift_vectors = new ArrayList<Buffer>();
		for (int i = 0; i < m_number_of_curves; ++i)
		{
			Buffer rnd_buffer = new Buffer(m_context, m_dataset.numAttributes()* DirectMemory.INT_SIZE);
			int[] random_vector = new int[dataset.numAttributes()];
			for (int j = 0; j < random_vector.length ; ++j)
			{
				random_vector[j] = rnd.nextInt(1000000);
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
			m_result_buffer = new Buffer(m_context, data.rows() * DirectMemory.DOUBLE_SIZE);
			m_result_index_buffer = new Buffer(m_context, data.rows() * DirectMemory.INT_SIZE);
			m_transform = new ZOrderTransform(m_context, instance.dataset().numAttributes(), data.rows());
			m_sort = new DoubleMergeSort(m_context, data.rows());
			m_ops = new Operations(m_context);
		}

		if (m_z_orders == null)
		{
			System.out.println("Building new z-order");
			m_min_max.fullMinMaxDouble(m_dataset, data, m_min_values, m_max_values);
			m_z_orders = new ArrayList<ZOrderItem[]>();
			m_transform.fillNormalizedData(instance.numAttributes(), data, m_min_values, m_max_values, m_attribute_types, true);
     
			for (int i = 0; i < m_number_of_curves; ++i)
			{
				ZOrderItem[] items = m_transform.createRandomShiftZOrder(m_random_shift_vectors.get(i), instance.dataset(),m_min_values, m_max_values, m_attribute_types, true);
				m_z_orders.add(items);
			}
      System.out.println("New z-order done");
			m_dirty = false;
		}
    //System.out.println("Evaluating");
		m_min_values.copyTo(m_min_values_with_test_instance);
		m_max_values.copyTo(m_max_values_with_test_instance);
		m_test_instance.begin(Buffer.WRITE);
		m_test_instance.set(instance, 0);
		m_test_instance.commit();
		m_min_max.updateMinMaxDouble(m_dataset, m_test_instance, m_min_values_with_test_instance, m_max_values_with_test_instance);
		HashSet<Integer> possible_candidates = new HashSet<Integer>();
		int index = 0;
		int strange = 0;
		for (ZOrderItem[] list : m_z_orders)
		{
      //System.out.println("Next item");
  
			byte[] code = m_transform.produceRandomShiftMortonCode( m_random_shift_vectors.get(index++), instance.dataset(),m_test_instance.attributes(), m_min_values, m_max_values, m_attribute_types, 1);
			ZOrderItem item = new ZOrderItem(code, 0, -1, (int)(instance.numAttributes() * DirectMemory.INT_SIZE ));
			
			
			int position = Arrays.binarySearch(list, item);
			if (position < 0) 
				position = -position-1;
			
		/*System.out.println("----------------> " );
			for (int i = 0 ;i < list.length ; ++i)
			{
				System.out.println(i+ " == "+ list[i]);
			}
			System.out.println("<----------------> " );*/ 
//			if (position == list.length || position == 0)
//				continue;
			int min = Math.max(position - K, 0);
			int max = Math.min(position + K, list.length); 
      
      
			ZOrderItem[] candidates = new ZOrderItem[max - min];
			System.arraycopy(list, min, candidates, 0, max - min);
			/*System.out.println("Looking for " + instance);
			data.begin(Buffer.READ);
			for (ZOrderItem it : candidates)
			{
				int idx = it.instanceIndex();
				Instance my_instance = data.read(idx, m_dataset);
				System.out.print(my_instance.toString() + " ");it.print();
			}
			data.commit();
			*/
			
			for (ZOrderItem next: candidates)
				possible_candidates.add(next.instanceIndex());
 
		}
    //System.out.println("Flush candidates");
		int[] cnd_items = new int[possible_candidates.size()];
		index = 0;
		for (Integer i : possible_candidates)
			cnd_items[index++] = i;
/*    System.out.print("Candidate ");
    for (int i : cnd_items)
    {
      System.out.print(i+ " ");
    }
    System.out.println();*/
		if (m_candidates_buffer == null) 
			m_candidates_buffer = new Buffer(m_context, data.rows() * DirectMemory.INT_SIZE);
		
		m_candidates_buffer.mapBuffer(Buffer.WRITE);
		m_candidates_buffer.writeArray(0, cnd_items);
		m_candidates_buffer.commitBuffer();
		
    
    
/*		double[] test_instance = new double[ m_dataset.numAttributes()];
		m_test_instance.attributes().mapBuffer(Buffer.READ);
		m_test_instance.attributes().readArray(0,test_instance );
		m_test_instance.attributes().commitBuffer();
		
		double[] values = new double[(int)(data.attributes().byteSize() / DirectMemory.DOUBLE_SIZE)];
		data.attributes().mapBuffer(Buffer.READ);
		data.attributes().readArray(0, values);
		data.attributes().commitBuffer();
		
		double[] min = new double[(int)(m_min_values_with_test_instance.byteSize()/DirectMemory.DOUBLE_SIZE)];
		m_min_values_with_test_instance.mapBuffer(Buffer.READ);
		m_min_values_with_test_instance.readArray(0, min);
		m_min_values_with_test_instance.commitBuffer();

		
		double[] max = new double[(int)(m_max_values_with_test_instance.byteSize()/DirectMemory.DOUBLE_SIZE)];
		m_max_values_with_test_instance.mapBuffer(Buffer.READ);
		m_max_values_with_test_instance.readArray(0, max);
		m_max_values_with_test_instance.commitBuffer();
*/        
    //System.out.println("Do short list");
    
    
		int sort_size = cnd_items.length;
		m_distance.squareDistance(
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
		m_sort.sort(m_result_buffer, m_result_index_buffer, sort_size);
		int[] nearest_k = new int[K];
		m_result_index_buffer.mapBuffer(Buffer.READ, 0, K * DirectMemory.INT_SIZE);
		m_result_index_buffer.readArray(0, nearest_k);
		m_result_index_buffer.commitBuffer();
		
		/*System.out.println("final selection " + instance);
		data.begin(Buffer.READ);
		for (int i = 0; i < nearest_k.length ; ++i)
		{
			Instance my_instance = data.read(cnd_items[nearest_k[i]], m_dataset);
			System.out.println(my_instance.toString() + " ");
		}
		data.commit();*/
		
		
		if (m_compute_approximation_error)
		{
      //System.out.println("Compute error");
			m_result_buffer.mapBuffer(Buffer.READ);
			double kth_approx = m_result_buffer.read(K * DirectMemory.DOUBLE_SIZE);
			m_result_buffer.commitBuffer();
			
			m_distance.squareDistance(m_dataset, 
					m_test_instance, 
					data, 
					m_min_values_with_test_instance, 
					m_max_values_with_test_instance, 
					m_attribute_types, 
					m_result_buffer);
			m_ops.prepareOrderKey(m_result_index_buffer, data.rows());
			m_sort.sort(m_result_buffer, m_result_index_buffer, data.rows());
			double kth_true;
			m_result_buffer.mapBuffer(Buffer.READ);
			kth_true = m_result_buffer.read(K * DirectMemory.DOUBLE_SIZE);
			
			m_result_buffer.commitBuffer();
			double eps = kth_approx/kth_true;
			ratio += eps;
			total_instances++;
			System.out.println("Current eps = " + eps + " avg "+ ratio/total_instances);
		
		}
		//System.out.println("Done");
		return makeDistribution(nearest_k, cnd_items, m_result_buffer, K);
    
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
