package moa.classifiers.gpu.zorder;

import java.util.Arrays;
import java.util.Random;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.opencl.util.CLogsSort;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.moa.opencl.util.MortonCode;
import org.moa.opencl.util.NarySearch;
import org.moa.opencl.util.Operations;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.core.Instances;


public class ZOrderTransform {
	
	private MortonCode m_morton_code;
	private CLogsVarKeyJava m_sort;
	private int m_src_dimensions;
	private int m_rows;
	private Buffer m_morton_code_buffer;
	private Buffer m_data_point_buffer;
	private Buffer m_normalized_data;
	private Operations m_operations;
	private Buffer m_sorted_code_order;
	private Buffer m_random_shift;
	private Random m_random;
	private Context m_context;
	
	
	public ZOrderTransform(Context ctx, int dimensions, int rows)
	{
		m_src_dimensions = dimensions;
		m_operations = new Operations(ctx);
		m_morton_code = new MortonCode(ctx, dimensions);
		m_sort = new CLogsVarKeyJava(ctx, false);
		m_rows = rows;
		m_normalized_data = new Buffer(ctx, rows*dimensions*DirectMemory.FLOAT_SIZE);
		m_data_point_buffer = new Buffer(ctx, rows * dimensions * DirectMemory.INT_SIZE);
		m_morton_code_buffer = new Buffer(ctx, rows * dimensions * DirectMemory.INT_SIZE);
		m_sorted_code_order = new Buffer(ctx, rows*DirectMemory.INT_SIZE);
		m_random_shift = new Buffer(ctx, dimensions* DirectMemory.INT_SIZE);
		m_random = new Random(System.currentTimeMillis());
		m_context = ctx;
	}
	
	
	
	public synchronized ZOrderItem[] findInZOrder(
			Instances dataset,
			ZOrderItem[] z_order, 
			Buffer instance,
			Buffer min_values, 
			Buffer max_values, 
			Buffer attribute_map, 
			int K)
	{
		int position = findPosition(dataset, z_order, instance, min_values, max_values, attribute_map);
		if (position < 0) 
			position = -position-1;
		int min = Math.max(position - K, 0);
		int max = Math.min(position + K, z_order.length);
		ZOrderItem[] candidates = new ZOrderItem[max - min];
		System.arraycopy(z_order, min, candidates, 0, max - min);
		return candidates;
	}

	public int findPosition(Instances dataset, ZOrderItem[] z_order, Buffer instance, Buffer min_values,
			Buffer max_values, Buffer attribute_map) {
		byte[] code = produceMortonCode(dataset, instance, min_values, max_values, attribute_map, 1);
		ZOrderItem item = new ZOrderItem(code, 0, -1, (int)(dataset.numAttributes() * DirectMemory.INT_SIZE ));
		int position = Arrays.binarySearch(z_order, item);
		return position;
	}

	public byte[] produceMortonCode(Instances dataset, Buffer instance, Buffer min_values, Buffer max_values,
			Buffer attribute_map, int rows) {
		m_operations.normalize(instance, m_normalized_data, 
				min_values, max_values, attribute_map, 
				dataset.numAttributes(), rows);
		m_operations.doubleToInt32(m_normalized_data, attribute_map, m_data_point_buffer,rows, dataset.numAttributes());
		
		m_morton_code.computeMortonCode(m_morton_code_buffer, m_data_point_buffer, rows);
        byte[] code = new byte[(int)(rows*dataset.numAttributes()*DirectMemory.INT_SIZE)];
        m_morton_code_buffer.mapBuffer(Buffer.READ, 0, rows * dataset.numAttributes() * DirectMemory.INT_SIZE);
        m_morton_code_buffer.readArray(0,code);
        m_morton_code_buffer.commitBuffer();
        return code;
	}
	
	public byte[] produceRandomShiftMortonCode(Buffer random_shift, Instances dataset, Buffer instance, Buffer min_values, Buffer max_values,
			Buffer attribute_map, int rows) {
		m_operations.normalize(instance, m_normalized_data, 
				min_values, max_values, attribute_map, 
				dataset.numAttributes(), rows);
		m_operations.doubleToInt32(m_normalized_data,attribute_map,  m_data_point_buffer,rows, dataset.numAttributes());
		m_operations.shiftByRandomVector(m_data_point_buffer, random_shift, dataset.numAttributes() , m_rows);
		m_morton_code.computeMortonCode(m_morton_code_buffer, m_data_point_buffer, rows);
        byte[] code = new byte[(int)(rows*dataset.numAttributes()*DirectMemory.INT_SIZE)];
        m_morton_code_buffer.mapBuffer(Buffer.READ, 0, rows * dataset.numAttributes() * DirectMemory.INT_SIZE);
        m_morton_code_buffer.readArray(0,code);
        m_morton_code_buffer.commitBuffer();
        return code;
	}
	
	public int[] canidatesForInstance(NarySearch search, ZOrderSequence sequence, Buffer random_shift, Instances dataset, Buffer instance, Buffer min_values, Buffer max_values,
			Buffer attribute_map,  int K) {
		m_operations.normalize(instance, m_normalized_data, 
				min_values, max_values, attribute_map, 
				dataset.numAttributes(), 1);
		m_operations.doubleToInt32(m_normalized_data,attribute_map,  m_data_point_buffer,1, dataset.numAttributes());
		m_operations.shiftByRandomVector(m_data_point_buffer, random_shift, dataset.numAttributes() , m_rows);
		m_morton_code.computeMortonCode(m_morton_code_buffer, m_data_point_buffer, 1);
		
		search.search(sequence.code(), sequence.indices(), m_morton_code_buffer, (int)(dataset.numAttributes() * DirectMemory.INT_SIZE) ,
				m_rows, false);
		int pos = search.getSearchPos();
		int min = Math.max(0,  pos - K);
		int max = Math.min(m_rows, pos +K);
		int[] candidates = new int[max - min];
		sequence.indices().mapBuffer(Buffer.READ, min*DirectMemory.INT_SIZE, (max - min)*DirectMemory.INT_SIZE);
		sequence.indices().readArray(0, candidates);
		sequence.indices().commitBuffer();
    return candidates;
	}
	
	
	public synchronized  ZOrderSequence createDeviceRandomShiftZOrder(Buffer random_shift, int num_attributes, 
			Buffer min_values, 
			Buffer max_values, 
			Buffer attribute_map, 
			boolean normalize
			)
	{
		m_normalized_data.copyTo(m_data_point_buffer);
		m_operations.shiftByRandomVector(m_data_point_buffer, random_shift,num_attributes , m_rows);
		m_morton_code.computeMortonCode(m_morton_code_buffer, m_data_point_buffer, m_rows);
		m_operations.prepareOrderKey(m_sorted_code_order, m_rows);
		m_sort.sort(m_sorted_code_order, m_morton_code_buffer, null, (int)(m_src_dimensions*DirectMemory.INT_SIZE), m_rows);
		ZOrderSequence seq = new ZOrderSequence(m_context,m_sorted_code_order, m_morton_code_buffer );
		return seq;
	}

	
	

	public synchronized  ZOrderItem[] createRandomShiftZOrder(Buffer random_shift, Instances dataset, 
			Buffer min_values, 
			Buffer max_values, 
			Buffer attribute_map, 
			boolean normalize
			)
	{

	
		m_operations.doubleToInt32(m_normalized_data, attribute_map, m_data_point_buffer, m_rows, dataset.numAttributes());
		
		m_operations.shiftByRandomVector(m_data_point_buffer, random_shift, dataset.numAttributes() , m_rows);
		m_morton_code.computeMortonCode(m_morton_code_buffer, m_data_point_buffer, m_rows);
		
		
		m_operations.prepareOrderKey(m_sorted_code_order, m_rows);
    
		m_sort.sort(m_sorted_code_order, m_morton_code_buffer, null, (int)(m_src_dimensions*DirectMemory.INT_SIZE), m_rows);
		
		byte[] morton_keys = new byte[(int)(m_rows* m_src_dimensions*DirectMemory.INT_SIZE)];
		m_morton_code_buffer.mapBuffer(Buffer.READ);
		m_morton_code_buffer.readArray(0, morton_keys);
		m_morton_code_buffer.commitBuffer();
		
		int[] morton_key_positions = new int[m_rows];
		m_sorted_code_order.mapBuffer(Buffer.READ);
		m_sorted_code_order.readArray(0, morton_key_positions);
		m_sorted_code_order.commitBuffer();
		
		ZOrderItem[] items = new ZOrderItem[ morton_key_positions.length ];
		for (int i = 0; i < items.length ; ++i)
		{
			items[i] = new ZOrderItem(morton_keys, (int)(morton_key_positions[i] * m_src_dimensions*DirectMemory.INT_SIZE), morton_key_positions[i], (int)(m_src_dimensions*DirectMemory.INT_SIZE));
		}
		return items;
	}
  
  


	public void fillNormalizedData(int num_attributes, DenseInstanceBuffer instances, Buffer min_values,
			Buffer max_values, Buffer attribute_map, boolean normalize) {
		assert(instances.rows() == m_rows);

		if (normalize)
		{
			m_operations.normalizeFloat(instances.attributes(), m_normalized_data, 
					min_values, max_values, attribute_map, 
					num_attributes, instances.rows());
		}
		else
		{
			instances.attributes().copyTo(m_normalized_data);
		}
	}
	
	
}
