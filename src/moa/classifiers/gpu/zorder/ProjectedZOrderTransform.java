package moa.classifiers.gpu.zorder;

import java.util.Arrays;
import java.util.Random;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.FJLT;
import org.moa.opencl.util.CLogsSort;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.moa.opencl.util.MortonCode;
import org.moa.opencl.util.NarySearch;
import org.moa.opencl.util.Operations;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.core.Instances;

public class ProjectedZOrderTransform {

	private MortonCode m_morton_code;
	private CLogsVarKeyJava m_sort;

	private int m_rows;
	private Buffer m_morton_code_buffer;
	private Buffer m_data_point_buffer;
	private Buffer m_normalized_data;
	private Operations m_operations;
	private Buffer m_sorted_code_order;
	private Buffer m_random_shift;
	private Buffer m_temp_result;
	private Random m_random;
	private Context m_context;
	private FJLT m_fjlt;
	private final int m_dst_dimensions;

	public ProjectedZOrderTransform(Context ctx, int dimensions, int rows, int target_dim) {
	
		m_dst_dimensions = target_dim;
		m_operations = new Operations(ctx);
		m_morton_code = new MortonCode(ctx, target_dim);
		m_sort = new CLogsVarKeyJava(ctx, false);
		m_rows = rows;
		m_fjlt = new FJLT(ctx, dimensions, target_dim);
		m_temp_result = new Buffer(ctx, target_dim * DirectMemory.DOUBLE_SIZE);
		m_normalized_data = new Buffer(ctx, rows * target_dim * DirectMemory.DOUBLE_SIZE);
		m_data_point_buffer = new Buffer(ctx, rows * target_dim * DirectMemory.INT_SIZE);
		m_morton_code_buffer = new Buffer(ctx, rows * target_dim * DirectMemory.INT_SIZE);
		m_sorted_code_order = new Buffer(ctx, rows * DirectMemory.INT_SIZE);
		m_random_shift = new Buffer(ctx, dimensions * DirectMemory.INT_SIZE);
		m_random = new Random(System.currentTimeMillis());
		m_context = ctx;
	}

	public int[] canidatesForInstance(NarySearch search, ZOrderSequence sequence, Buffer random_shift,
			Buffer instance, Buffer min_values, Buffer max_values, Buffer attribute_map, int K) {
		m_fjlt.transform(instance, m_normalized_data);
		m_operations.doubleToInt32(m_normalized_data, attribute_map, m_data_point_buffer, 1, m_dst_dimensions);
		m_operations.shiftByRandomVector(m_data_point_buffer, random_shift, m_dst_dimensions, m_rows);
		m_morton_code.computeMortonCode(m_morton_code_buffer, m_data_point_buffer, 1);

		search.search(sequence.code(), sequence.indices(), m_morton_code_buffer,
				(int) (m_dst_dimensions * DirectMemory.INT_SIZE), m_rows, false);
		int pos = search.getSearchPos();
		
		int min = Math.max(0, pos - m_rows/4);
		int max = Math.min(m_rows, pos +m_rows/4);
		int[] candidates = new int[max - min];
		sequence.indices().mapBuffer(Buffer.READ, min * DirectMemory.INT_SIZE, (max - min) * DirectMemory.INT_SIZE);
		sequence.indices().readArray(0, candidates);
		sequence.indices().commitBuffer();
		return candidates;
	}

	public synchronized ZOrderSequence createDeviceRandomShiftZOrder(Buffer random_shift, 
			 Buffer attribute_map, boolean normalize) {
		m_operations.doubleToInt32(m_normalized_data, attribute_map, m_data_point_buffer, m_rows, m_dst_dimensions);
		m_operations.shiftByRandomVector(m_data_point_buffer, random_shift, m_dst_dimensions, m_rows);
		m_morton_code.computeMortonCode(m_morton_code_buffer, m_data_point_buffer, m_rows);
		m_operations.prepareOrderKey(m_sorted_code_order, m_rows);
		m_sort.sort(m_sorted_code_order, m_morton_code_buffer, null, (int) (m_dst_dimensions * DirectMemory.INT_SIZE),
				m_rows);
		ZOrderSequence seq = new ZOrderSequence(m_context, m_sorted_code_order, m_morton_code_buffer);
		return seq;
	}

	public void fillNormalizedData(int num_attributes, DenseInstanceBuffer instances, Buffer min_values,
			Buffer max_values, Buffer attribute_map, boolean normalize) {
		assert (instances.rows() == m_rows);

		m_fjlt.transform(instances.attributes(), instances.rows(), m_normalized_data);
		/*
		 * if (normalize) { m_operations.normalize(instances.attributes(),
		 * m_normalized_data, min_values, max_values, attribute_map,
		 * num_attributes, instances.rows()); } else {
		 * instances.attributes().copyTo(m_normalized_data); }
		 */
	}

}
