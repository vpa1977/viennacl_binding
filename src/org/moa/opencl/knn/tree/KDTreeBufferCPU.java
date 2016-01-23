package org.moa.opencl.knn.tree;

import org.moa.gpu.DenseInstanceBuffer;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.core.Instances;

/**
 * data structure -> [ ][ ] - plain array of instances [1][2][3][4][5][6][7][8]
 * - array of instance indices [node_id][node_id] - array of node _ids
 * [split_dim] - dimension for each node [split_value] - split value for each
 * node We can either do a median split - 1) find widest dimension 2) sort
 * according to widest dimension - N sort calls 3) update node indices
 * 
 * @author john
 *
 */
public class KDTreeBufferCPU {
	private Instances m_dataset;
	private DenseInstanceBuffer m_instance_data;
	private int[] m_leaf_node_ids;
	private int[] m_node_split_dim;
	private double[] m_node_split_value;
	private double[] m_max_temp;
	private double[] m_min_temp;
	
	private int m_branch_nodes = 0;
	private int m_num_attributes;
	private int m_tree_depth;

	public KDTreeBufferCPU(Instances dataset, Context ctx, int tree_depth, DenseInstanceBuffer buffer) {
		m_instance_data = buffer;
		m_tree_depth = tree_depth;
		m_leaf_node_ids = new int[buffer.rows()];
		
		m_branch_nodes = (int) (Math.pow(2, tree_depth + 1) - 1) - (int)Math.pow(2, tree_depth);
		m_dataset = dataset;
		m_num_attributes = dataset.numAttributes();

		m_max_temp = new double[m_dataset.numAttributes() *  m_branch_nodes];
		m_min_temp = new double[m_dataset.numAttributes() *  m_branch_nodes];
		m_node_split_dim  = new int[m_branch_nodes];
		m_node_split_value = new double[m_branch_nodes];
	}

	private long nextPow2(long v) {
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v++;
		return v;
	}

	public long level(long node_num) {
		if (node_num == 0)
			return 0;
		long next = nextPow2(node_num);
		long kplusone = (long) (Math.log(next) / Math.log(2));
		return kplusone - 1;
	}

	public long max_level_id(long level) {
		if (level == 0)
			return 0;
		return (int) Math.pow(2, level + 1) - 2;
	}

	public long child(long id) {
		long level = level(id);
		long max_prev_level = max_level_id(level - 1);
		long max_cur_level = max_level_id(level);
		long count = id - max_prev_level;
		return count * 2 + max_cur_level - 1;
	}

	/**
	 * CPU build implementation
	 * 
	 * @param level
	 */
	public void buildCPU(int level) {
		int prev_level = level - 1;
		long start_node = 0;
		long end_node =0;
		if (level > 1)
		{
			start_node = max_level_id(level - 2)+1;
			end_node = max_level_id(level -1);
		}

		m_instance_data.begin(Buffer.READ_WRITE);
		// find min_max
		for (int i = 0; i < m_instance_data.rows(); ++i)
		{
			int node_id = m_leaf_node_ids[i];
			if (!(node_id >= start_node && node_id <= end_node))
				throw new RuntimeException("All instances should be in the leaves");
			for (int att = 0; att < m_num_attributes; ++att)
			{
				int node_offset = node_id * m_num_attributes + att;
				Buffer attributes = m_instance_data.attributes();
				double value = attributes.read((att + i * m_num_attributes) * DirectMemory.DOUBLE_SIZE);
				double min =m_min_temp[node_offset];
				double max =m_max_temp[node_offset];
				if (value > max)
					m_max_temp[node_offset]= value;
				if (value < min)
					m_min_temp[node_offset]= value;
			}
		}
		
		// split by mean of widest dim
		int split_candidate = -1;
		double max_range = -1;
		
		for (int node = (int)start_node; node <= end_node; ++node)
		{
			// 1 thread
			for (int att = 0; att < m_num_attributes; ++att)
			{
				double min_val = m_min_temp[(node * m_num_attributes + att)];
				double max_val = m_max_temp[(node * m_num_attributes + att)];
				
				double abs_min_val = m_min_temp[att];
				double abs_max_val = m_max_temp[att];
				double range = (abs_max_val - abs_min_val);
				if (range > 0)
				{
					double node_range = (max_val - min_val)/range;
					if (node_range > max_range)
					{
						split_candidate = att;
						max_range = node_range;
					}
				}
			}
			m_node_split_dim[node]=  split_candidate;
			m_node_split_value[node]= m_min_temp[(node * m_num_attributes + split_candidate)] + max_range/2;
		}
		
		
		for (int i = 0; i < m_instance_data.rows(); ++i)
		{
			int node_id = m_leaf_node_ids[i];
			int split_dim  = m_node_split_dim[node_id];
			double split_value = m_node_split_value[node_id];
			
			double att_value = m_instance_data.attributes().read( (split_dim + i * m_num_attributes) * DirectMemory.DOUBLE_SIZE);
			if (att_value > split_value)
				m_leaf_node_ids[i]= (int) child(node_id)+1;
			else
				m_leaf_node_ids[i]= (int) child(node_id);
			
		}
		
		m_instance_data.commit();
		
	}
	
	public void reorder()
	{
		long start_node = max_level_id(m_tree_depth - 1)+1;
		long end_node = max_level_id(m_tree_depth);
	}

	public DenseInstanceBuffer getInstanceData() {
		return m_instance_data;
	}

	public void setInstanceData(DenseInstanceBuffer m_instance_data) {
		this.m_instance_data = m_instance_data;
	}

	public void prepareLevel0() {
		m_instance_data.begin(Buffer.READ_WRITE);
		Buffer attributes = m_instance_data.attributes();
		
		// init min max
		for (int i = 0 ; i < m_leaf_node_ids.length ; ++i)
			m_leaf_node_ids[i] = 0;
		for (int i = 0; i < m_min_temp.length; ++i)
			m_min_temp[i]= Double.MAX_VALUE;
		for (int i = 0; i < m_max_temp.length; ++i)
			m_max_temp[i]= -Double.MAX_VALUE;
		
		
	/*	for (int i = 0; i < m_instance_data.rows(); ++i)
		{
			for (int att = 0; att < m_num_attributes; ++att)
			{
				int node_offset = (int)( att );
				
				double value = attributes.read( (att + i * m_num_attributes) * DirectMemory.DOUBLE_SIZE);
				System.out.println("Att " + att + " value "+ value);
				double min =m_min_temp[node_offset];
				double max =m_max_temp[node_offset];
				if (value > max)
					m_max_temp[node_offset]= value;
				if (value < min)
					m_min_temp[node_offset]= value;
			}
		}
		*/
		m_instance_data.commit();
	}

	public void dumpTree() {
		for (int level = 0; level <= m_tree_depth; ++level)
		{
			long start_node = 0;
			long end_node =0;
			if (level > 0)
			{
				start_node = max_level_id(level - 2)+1;
				end_node = max_level_id(level -1);
			}
			for (int node = (int)start_node; node <= end_node; ++node)
			{
				int split_dim = m_node_split_dim[node];
				double split = m_node_split_value[node];
				System.out.print(node + "("+split_dim + ","+ split +") [");
			/*	for (int att = 0; att < m_num_attributes ; ++ att)
				{
					double min = m_min_temp[(att + node * m_num_attributes)];
					double max = m_max_temp[(att + node * m_num_attributes)];
					System.out.print("(" + min + "," + max + ")");
				}
				*/
				System.out.print("]");
				
			}
			System.out.println();
		}
		
	}
}
