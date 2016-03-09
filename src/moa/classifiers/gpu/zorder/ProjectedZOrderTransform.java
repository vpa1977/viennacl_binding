package moa.classifiers.gpu.zorder;

import java.math.BigInteger;
import java.util.Arrays;
import java.util.Random;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.FJLT;
import org.moa.gpu.FloatFJLT;
import org.moa.gpu.MatrixRandomProjection;
import org.moa.opencl.util.BufHelper;
import org.moa.opencl.util.CLogsSort;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.moa.opencl.util.MinMax;
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
	private FloatFJLT m_fjlt;
	private MatrixRandomProjection m_proj;
	private MinMax m_min_max;
	private final int m_dst_dimensions;
	private Buffer m_max_values;
	private Buffer m_min_values;
	private Buffer m_attr_map;
	private Buffer projected_instance;
	private Buffer projected_norm_instance;
	private Buffer projected_norm_code;
	private NarySearch m_search;

	public ProjectedZOrderTransform(Context ctx, CLogsVarKeyJava sorter,  int dimensions, int rows, int target_dim) {
		
		m_context = ctx;
		m_dst_dimensions = target_dim;
		m_operations = new Operations(ctx);
		m_morton_code = new MortonCode(ctx, target_dim);
		m_sort =sorter;
		m_rows = rows;
		//m_fjlt = new FloatFJLT(ctx, target_dim, dimensions );
		m_min_max = new MinMax(ctx);
		m_min_values = new Buffer(m_context, m_dst_dimensions * DirectMemory.FLOAT_SIZE);
		m_max_values = new Buffer(m_context, m_dst_dimensions * DirectMemory.FLOAT_SIZE);
		m_attr_map = new Buffer(m_context, m_dst_dimensions * DirectMemory.INT_SIZE);
		m_attr_map.fill((byte)0);
		m_temp_result = new Buffer(ctx, target_dim * DirectMemory.FLOAT_SIZE);
		m_normalized_data = new Buffer(ctx, rows * target_dim * DirectMemory.FLOAT_SIZE);
		m_data_point_buffer = new Buffer(ctx, rows * target_dim * DirectMemory.FLOAT_SIZE);
		m_morton_code_buffer = new Buffer(ctx, rows * target_dim * DirectMemory.INT_SIZE);
		m_sorted_code_order = new Buffer(ctx, rows * DirectMemory.INT_SIZE);
		m_random_shift = new Buffer(ctx, dimensions * DirectMemory.INT_SIZE);
		m_random = new Random(System.currentTimeMillis());
		m_proj = new MatrixRandomProjection(m_context, target_dim, dimensions);
		m_search = new NarySearch(m_context, false);
		projected_instance = new Buffer(m_context, m_dst_dimensions * DirectMemory.FLOAT_SIZE);
		projected_norm_instance = new Buffer(m_context, m_dst_dimensions * DirectMemory.FLOAT_SIZE);
		projected_norm_code = new Buffer(m_context, m_dst_dimensions * DirectMemory.FLOAT_SIZE);

	}
	
	public void printCurve()
	{
		int[] sorted_order = BufHelper.rbi(m_sorted_code_order);
		byte[] code = BufHelper.rbb(m_morton_code_buffer);
		int code_len =(int)(m_dst_dimensions * DirectMemory.INT_SIZE);
		for (int i = 0 ; i< sorted_order.length; ++i)
		{
			byte[] single_code = new byte[code_len];
			System.arraycopy(code, sorted_order[i]*code_len, single_code, 0, code_len);
			System.out.print(i + "\t");
			printCode(single_code);
		}
	}
	  public static String byteToString(byte b) {
		    byte[] masks = { -128, 64, 32, 16, 8, 4, 2, 1 };
		    StringBuilder builder = new StringBuilder();
		    for (byte m : masks) {
		        if ((b & m) == m) {
		            builder.append('1');
		        } else {
		            builder.append('0');
		        }
		    }
		    return builder.toString();
		}
	void printCode(byte[] code)
	{
		for (byte b :code)
		{
			String s = Integer.toHexString(b);
			if (s.length() != 2)
				s = '0'+s;
			System.out.print(s);
		}
		System.out.println();
		
	}
	
	public void enqueueSearch(Buffer random_shift,	Buffer instance)
	{
		m_proj.project(instance, projected_instance);
		m_operations.normalizeFloatReplaceMin(projected_instance, projected_norm_instance, m_min_values, m_max_values, m_attr_map, m_dst_dimensions, 1);
		//printCurve();
		m_morton_code.computeMortonCode(projected_norm_code, projected_norm_instance, 1);
		//printCode(BufHelper.rbb(projected_norm_code));
		
		m_search.search(m_morton_code_buffer, m_sorted_code_order, projected_norm_code,
				(int) (m_dst_dimensions * DirectMemory.INT_SIZE), m_rows, true);
	}

	public int[] canidatesForInstance(int K) {
		int pos = m_search.getSearchPos();
		//System.out.println("Pos " + pos);
		int min = Math.max(0, pos - 512);
		int max = Math.min(m_rows, pos +512);
		int[] candidates = new int[max - min];
		m_sorted_code_order.mapBuffer(Buffer.READ, min * DirectMemory.INT_SIZE, (max - min) * DirectMemory.INT_SIZE);
		m_sorted_code_order.readArray(0, candidates);
		m_sorted_code_order.commitBuffer();
		return candidates;
	}

	public synchronized void createDeviceRandomShiftZOrder( Buffer random_shift) {
		//m_operations.shiftByRandomVector(m_normalized_data, random_shift, m_dst_dimensions, m_rows);
		m_morton_code.computeMortonCode(m_morton_code_buffer, m_normalized_data, m_rows);
		m_operations.prepareOrderKey(m_sorted_code_order, m_rows);
		m_sort.sort(m_sorted_code_order, m_morton_code_buffer, null, (int) (m_dst_dimensions * DirectMemory.INT_SIZE),
				m_rows);
	}

	public void fillNormalizedData(Instances dataset, DenseInstanceBuffer instances) {
		assert (instances.rows() == m_rows);
		//instances.attributes().copyTo(m_normalized_data);
		m_proj.project(instances.attributes(), instances.rows(), m_data_point_buffer);
		m_min_max.fullMinMaxFloat(m_dst_dimensions, m_data_point_buffer,instances.rows(), m_min_values, m_max_values);
		m_operations.normalizeFloat(m_data_point_buffer, m_normalized_data, m_min_values, m_max_values, m_attr_map, m_dst_dimensions, m_rows);
	}

}
