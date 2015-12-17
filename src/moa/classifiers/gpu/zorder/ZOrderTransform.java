package moa.classifiers.gpu.zorder;

import java.util.Arrays;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.opencl.util.CLogsSort;
import org.moa.opencl.util.MortonCode;
import org.moa.opencl.util.Operations;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.core.Instances;


public class ZOrderTransform {
	
	private MortonCode m_morton_code;
	private CLogsSort m_sort;
	private int m_src_dimensions;
	private int m_rows;
	private Buffer m_morton_code_buffer;
	private Buffer m_data_point_buffer;
	private Buffer m_normalized_data;
	private Operations m_operations;
	private Buffer m_temp_sort_buffer;
	private Buffer m_sorted_code_order;
	
	
	public ZOrderTransform(Context ctx, int dimensions, int rows)
	{
		m_src_dimensions = dimensions;
		m_operations = new Operations(ctx);
		m_morton_code = new MortonCode(ctx, dimensions);
		m_sort = new CLogsSort(ctx);
		m_rows = rows;
		m_normalized_data = new Buffer(ctx, rows*dimensions*DirectMemory.DOUBLE_SIZE);
		m_data_point_buffer = new Buffer(ctx, rows * dimensions * DirectMemory.INT_SIZE);
		m_morton_code_buffer = new Buffer(ctx, rows * dimensions * DirectMemory.INT_SIZE);
		m_temp_sort_buffer = new Buffer(ctx, rows*DirectMemory.INT_SIZE);
		m_sorted_code_order = new Buffer(ctx, rows*DirectMemory.INT_SIZE);
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
	/*	int dp = -1;
		for (int i = 0; i < z_order.length ; ++i)
			if (z_order[i].equals(item))
				dp = i;
		if (dp != position)
		{
			int res = z_order[dp].compareTo(item);
			System.out.println("!!");
		}
		*/	
		return position;
	}

	public byte[] produceMortonCode(Instances dataset, Buffer instance, Buffer min_values, Buffer max_values,
			Buffer attribute_map, int rows) {
		m_operations.normalize(instance, m_normalized_data, 
				min_values, max_values, attribute_map, 
				dataset.numAttributes(), rows);
		m_operations.doubleToInt32(m_normalized_data, m_data_point_buffer,rows, dataset.numAttributes());
		
		/*int[] data_points = new int[rows * dataset.numAttributes()];
		double[] normalized_data = new double[rows * dataset.numAttributes()];
		double[] before_norm = new double[rows * dataset.numAttributes()];
		
		instance.mapBuffer(Buffer.WRITE);
		instance.readArray(0, before_norm);
		instance.commitBuffer();
		m_normalized_data.mapBuffer(Buffer.WRITE);
		m_normalized_data.readArray(0, normalized_data);
		m_normalized_data.commitBuffer();
		m_data_point_buffer.mapBuffer(Buffer.READ);
		m_data_point_buffer.readArray(0, data_points);
		m_data_point_buffer.commitBuffer();
		*/
		
		m_morton_code.computeMortonCode(m_morton_code_buffer, m_data_point_buffer, rows);
		byte[] code = new byte[(int)(rows*dataset.numAttributes()*DirectMemory.INT_SIZE)];
		m_morton_code_buffer.mapBuffer(Buffer.READ, 0, rows * dataset.numAttributes() * DirectMemory.INT_SIZE);
		m_morton_code_buffer.readArray(0,code);
		m_morton_code_buffer.commitBuffer();
		return code;
	}
	

	public synchronized  ZOrderItem[] createZOrder(Instances dataset, 
			DenseInstanceBuffer instances,
			Buffer min_values, 
			Buffer max_values, 
			Buffer attribute_map, 
			boolean normalize
			)
	{
		assert(instances.rows() == m_rows);
		
		if (normalize)
		{
			m_operations.normalize(instances.attributes(), m_normalized_data, 
					min_values, max_values, attribute_map, 
					dataset.numAttributes(), instances.rows());
		}
		else
		{
			instances.attributes().copyTo(m_normalized_data);
		}
		m_operations.doubleToInt32(m_normalized_data, m_data_point_buffer, m_rows, dataset.numAttributes());
		m_morton_code.computeMortonCode(m_morton_code_buffer, m_data_point_buffer, m_rows);
		m_sort.sort(m_temp_sort_buffer, m_morton_code_buffer, m_sorted_code_order, (int)(m_src_dimensions*DirectMemory.INT_SIZE), m_rows);
		
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
	
	
}
