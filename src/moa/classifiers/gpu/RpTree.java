package moa.classifiers.gpu;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.FJLT;
import org.moa.opencl.util.BufHelper;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.moa.opencl.util.Distance;
import org.moa.opencl.util.MinMax;
import org.moa.opencl.util.Operations;
import org.moa.opencl.util.TreeUtil;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.lazy.neighboursearch.EuclideanDistance;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.FlagOption;
import moa.options.IntOption;
import moa.options.MultiChoiceOption;
import moa.tasks.TaskMonitor;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Random Projection Tree implementation
 * 
 * @inproceedings{freund2007learning, title={Learning the structure of manifolds
 *                                    using random projections}, author={Freund,
 *                                    Yoav and Dasgupta, Sanjoy and Kabra,
 *                                    Mayank and Verma, Nakul},
 *                                    booktitle={Advances in Neural Information
 *                                    Processing Systems}, pages={473--480},
 *                                    year={2007} } Translated into WEKA from
 *                                    author's implementation
 *                                    http://cseweb.ucsd.edu/~naverma/RPTrees/
 *
 */
public class RpTree extends AbstractClassifier {

	static {
		System.loadLibrary("viennacl-java-binding");
	}

	public FlagOption defeatistOption = new FlagOption("defeatist", 'f', "Use defeatist search");
	public IntOption kOption = new IntOption("k", 'k', "k-Nearest", 5);
	public IntOption windowSizeOption = new IntOption("windowSize", 'w', "Sliding Window Size", 65535);
	public IntOption maxTreeDepthOption = new IntOption("maxTreeDepth", 'd', "Max Tree Depth",8 );
	public IntOption numProjectionsOption = new IntOption("numProjections", 'n', "Number of projections", 20 );
	public MultiChoiceOption contextUsedOption = new MultiChoiceOption("contextUsed", 'c', "Context Type",
			new String[] { "CPU", "OPENCL", "HSA" }, new String[] { "CPU single thread",
					"OpenCL offload. Use OPENCL_DEVICE Env. variable to select device", "HSA Offload" },
			0);
	
	public MultiChoiceOption projectionMethodOption = new MultiChoiceOption("projectionMethod", 'p', "Projection method",
			new String[] { "CPU", "ViennaCL MM", "FJLT" }, new String[] { "CPU serial",
					"ViennaCL matrix mult", "FJLT" },
			0);

	private Context m_context;

	private Tree m_tree;

	private ArrayDeque<InstanceData> m_window;
	private int m_num_projections;

	private int m_depth;

	private class InstanceData {
		public InstanceData(Instance i, double[] projection) {
			m_instance = i;
			m_projection = projection;
		}

		Instance m_instance;
		double[] m_projection;
	}

	
	
	private HashMap<Integer, ArrayList<InstanceData>> m_buckets;

	private EuclideanDistance m_euclidean_distance;
	
	private Measurement m_number_of_distance_calcuations = new Measurement("Number of distance calcuations", 0);
	private Measurement m_total_instances_tested = new Measurement("Total instances tested", 0);
	private Distance m_gpu_distance;
	private MinMax m_min_max;


	
	private class RpNode {
		private int m_vect_length;
		private int m_num_projections;

		public RpNode(int node_index, boolean leaf, int vect_length, int num_projections) {
			this.node_ndx = node_index;
			this.is_leaf = leaf;
			this.m_vect_length = vect_length;
			this.m_num_projections = num_projections;
			clear();
		}

		public void clear() {
			this.total = 0;
			this.mean = new double[m_vect_length];
			this.proj_sum = new double[m_num_projections];
			this.proj_sum_sq = new double[m_num_projections];
			this.smallbin = new double[m_num_projections][Tree.NUM_BINS_SMALL];
			this.largebin = new double[m_num_projections][2 * Tree.NUM_BINS_LARGE];

		}

		int node_ndx;
		int total;
		boolean is_leaf;
		public int proj_type;
		public int proj_to_use;
		public double threshold;
		public double[] proj_sum;
		public double[] mean;
		public double[] proj_sum_sq;
		public double[][] smallbin;
		public double[][] largebin;
	}

	private class Tree {
		static final int NUM_BINS_SMALL = 8;
		static final int NUM_BINS_LARGE = 6;
		int m_decay_count = 65536;
		static final int N1 = 50;
		static final int N2 = 50;
		static final int N3 = 250;
		static final double prmE = 0.05;
		static final double prmC = 10;

		private int m_num_projections;
		private int m_vect_length;
		private ArrayList<double[]> m_projections;
		private ArrayList<RpNode> m_nodes;
		private int m_depth;
		private ArrayList<Integer> m_invalidatedBuckets;

		public Tree(int num_projections, int vect_length, int depth, int decay_count) {
			Random rnd = new Random(System.currentTimeMillis());
			m_depth = depth;
			m_decay_count = decay_count;
			m_projections = new ArrayList<double[]>();
			m_nodes = new ArrayList<RpNode>();
			m_num_projections = num_projections;
			m_vect_length = vect_length;

			
			m_gpu_projection_buffer.mapBuffer(Buffer.WRITE);
			for (int i = 0; i < m_num_projections; ++i) {
				double[] proj = new double[m_vect_length];
				for (int j = 0; j < m_vect_length; ++j) {
					proj[j] = rnd.nextDouble();
				}
				m_gpu_projection_buffer.writeArray(i * m_vect_length , proj);
				m_projections.add(proj);
			}
			m_gpu_projection_buffer.commitBuffer();
		
			
			for (int i = 0; i < (0x1 << m_depth) - 1; i++) {
				RpNode node = new RpNode(i, true, vect_length, num_projections);
				m_nodes.add(node);
			}
		}

		public void trainOnInstance(Instance inst, double[] projections) {
			// m_invalidatedBuckets = new ArrayList<Integer>();
			// prepare projections for each of the splits (complexity
			// Nprojections x d^2)
			//
			int ndx = 0;
			for (int lvl = 0; lvl < m_depth; lvl++) { // for each level in tree
				RpNode n = m_nodes.get(ndx);
				learnRpNode(n, inst, projections);
				if (n.total < N1 + N2)
					break;
				ndx = ndx * 2 + 1; // ready the index for the next iteration
				if (n.proj_type == 1) { // type 1 projection, ie median split
					if (projections[n.proj_to_use] > n.threshold)
						ndx++;
				} else if (n.proj_type == 2) { // type 2 projection, ie center
												// split
					if (Math.abs(projections[n.proj_to_use] - n.proj_sum[n.proj_to_use]) < n.threshold)
						ndx++;
				} else {
					throw new RuntimeException("Unrecognized projection type: " + n.proj_type);
				}

			} // loop
		}

		public int findNode(Instance inst, double[] projections) {

			int ndx = 0;
			for (int level = 0; level < m_depth; level++) {
				RpNode node = m_nodes.get(ndx);

				if (node.is_leaf)
					return ndx;

				double mean = node.proj_sum[node.proj_to_use];

				ndx = ndx * 2 + 1;
				if (node.proj_type == 1) { // median split
					if (projections[node.proj_to_use] > node.threshold) {
						ndx++;
					}
				} else if (node.proj_type == 2) { // center split
					if (Math.abs(projections[node.proj_to_use] - mean) < node.threshold) {
						ndx++;
					}
				} else {
					throw new RuntimeException("Unrecognized projection type");
				}
			}
			return ndx;
		}
		
		private void computeWekaWay(Instance inst, double[] projections)
		{
			for (int i =0; i < projections.length ; ++i)
			{
				for (int j = 0; j< inst.numAttributes(); ++j)
					projections[i] += inst.isMissing(j) ? 0 :  m_projections.get(i)[j] * inst.value(j);
			}
		}

		private void computeProjections(Instance inst, double[] projections) {
			
			
			switch  (projectionMethodOption.getChosenIndex())
			{
			case 0:
				computeWekaWay(inst, projections);
				break;
			case 1:
				computeProjectionAx(inst, projections);
				break;
			case 2:
				computeProjectionFJLT(inst, projections);
				break;
			}
		}

		private void computeProjectionAx(Instance i, double[] projections) {
			m_instance_buffer.begin(Buffer.WRITE);
			m_instance_buffer.set(i, 0);
			m_instance_buffer.commit();
			m_gpu_operations.dense_ax(m_gpu_projection_buffer, m_instance_buffer.attributes(), m_projection_result, m_num_projections, i.numAttributes());
			m_projection_result.mapBuffer(Buffer.READ);
			m_projection_result.readArray(0, projections);
			m_projection_result.commitBuffer();
		}

		private void computeProjectionFJLT(Instance inst, double[] projections) {
			int index = 0;
			m_source_transform_buffer.begin(Buffer.WRITE);
			m_source_transform_buffer.set(inst, 0);
			m_source_transform_buffer.commit();
			for (FJLT f : m_fjlt)
			{
				f.transform(m_source_transform_buffer.attributes(), m_target_buffer);
				double[] result = new double[(int)(m_target_buffer.byteSize()/DirectMemory.DOUBLE_SIZE)];
				m_target_buffer.mapBuffer(Buffer.READ);
				m_target_buffer.readArray(0, result);
				m_target_buffer.commitBuffer();
				System.arraycopy(result, 0, projections, index, Math.min(projections.length- index, result.length));
				index += result.length;
				if (index >= projections.length)
					break;
			}
		}

		private void learnRpNode(RpNode n, Instance inst, double[] projections) {
			double alpha; // decay parameter
			int i, p; // counter

			n.total++; // count update

			if (n.total > m_decay_count) {
				alpha = 1. / m_decay_count;
			} else {
				alpha = 1. / (n.total);
			}

			computeMeans(n, inst, alpha);

			p = update_projection_stats(n, projections, alpha);

			if (n.total == N1 + N2) { // sufficient points to have a stable
										// threshold
				threshold_split(n);
			}

			// TODO: every so often update the threshold if not in center split
			// if(((n->total+1) % (N1+N2) == 1 ) && (n->proj_type !=2))
			// { update_thresh(n, n->proj_to_use); }

			if (n.proj_type != 2) { // if not center split, check for center
									// split every so often
				if ((n.total - (N1 + N2) + 2) % N3 == 1) {
					center_split(n);
				}
			}

		}

		private int update_projection_stats(RpNode n, double[] projections, double alpha) {
			int p;
			for (p = 0; p < m_num_projections; p++) { // proj sum / sum_sq
														// update
				update_stats(n, projections[p], p, alpha);
				if (n.total > N1) { // histogram update
					update_hist(n, projections[p], p, N1);
				}
			}
			return p;
		}

		private void threshold_split(RpNode n) {
			int p;
			double maxx = 0, tmp, thresh = 0;
			int maxp = -1;
			for (p = 0; p < m_num_projections; p++) {
				tmp = update_thresh(n, p); // find the max gain proj
				if (tmp > maxx) {
					maxx = tmp;
					maxp = p;
					thresh = n.threshold;
				}
			}
			n.proj_to_use = maxp;
			n.threshold = thresh;
			n.proj_type = 1;
			n.is_leaf = false;
		}

		private void computeMeans(RpNode n, Instance inst, double alpha) {

			for (int i = 0; i < inst.numAttributes(); i++) { // mean update
				n.mean[i] = inst.isMissing(i) ? (1 - alpha) * (n.mean[i])
						: (1 - alpha) * (n.mean[i]) + inst.value(i) * alpha;
			}
		}

		private void center_split(RpNode n) {
			int i, j, p; // lcvs

			double threshc;
			int maxpc = -1;
			int toCenterSplit = 0, vlToUse = 0;

			for (p = 0; p < m_num_projections; p++) {
				double mean = n.proj_sum[p];
				double stddev = Math.sqrt(n.proj_sum_sq[p] - mean * mean);

				// check for center split
				for (i = 0; i < NUM_BINS_LARGE; i++) {
					double sum = 0;
					for (j = i; j < NUM_BINS_LARGE; j++) {
						sum += n.largebin[p][j * 2] + n.largebin[p][j * 2 + 1];
					}
					if (sum > Math.max(prmE, prmC * (1 - erf(i / Math.sqrt(2))))) {
						vlToUse = i;
						toCenterSplit = 1;
						break;
					}
				}

				if (toCenterSplit == 1) {
					maxpc = p;
					threshc = (vlToUse) * stddev;
					break; // using the first center split
				}
			}

			if (toCenterSplit == 1) {
				p = maxpc;
				double mean = n.proj_sum[p];
				double stddev = Math.sqrt(n.proj_sum_sq[p] - mean * mean);

				// reset the learnt parameters..
				reset_children(n.node_ndx);
				n.is_leaf = false;
				n.proj_type = 2;

				// find the threshold
				double sum = 0;
				for (i = 0; i < NUM_BINS_LARGE; i++)
					sum += n.largebin[p][i * 2] + n.largebin[p][i * 2 + 1];

				for (i = 0; i < NUM_BINS_SMALL; i++)
					sum += n.smallbin[p][i];

				double cur_ll = 0, max_ll = 1;
				int max_ll_ndx = -1;

				for (i = 1; i < ((int) (NUM_BINS_SMALL / 2) + NUM_BINS_LARGE - 2); i++) {
					cur_ll = logll(n, i, maxpc);
					if ((cur_ll > max_ll || max_ll > 0) && cur_ll < 0) {
						max_ll = cur_ll;
						max_ll_ndx = i;
					}
				}

				n.proj_to_use = maxpc;
				if (max_ll_ndx > (int) (NUM_BINS_SMALL / 2)) {
					n.threshold = (max_ll_ndx - (int) (NUM_BINS_SMALL / 2) + 1) * stddev;
				} else {
					n.threshold = ((double) max_ll_ndx) * 2 / NUM_BINS_SMALL * stddev;
				}
			}
		}

		//
		// internal function called to reset all the descendants of a
		// specific RPNode of a tree
		// Parameters: tree - pointer to the RPTree
		// parent_ndx - index of the parent node
		void reset_children(int parent_ndx) {
			int leftchild = 2 * parent_ndx + 1;
			int rightchild = 2 * parent_ndx + 2;

			if (rightchild < (0x1 << m_depth) - 1) {
				reset_node(m_nodes.get(leftchild));
				reset_node(m_nodes.get(rightchild));
				reset_children(leftchild);
				reset_children(rightchild);
			}
		}

		void reset_node(RpNode n) {
			n.clear();
			n.is_leaf = true;
			//m_invalidatedBuckets.add(n.node_ndx);
		}

		class Stats {
			double mean1;
			double mean2;
			double stddev1;
			double stddev2;
		}

		//
		// internal function to calculate the log likelihood
		//
		double logll(RpNode n, int threshold, int p) {

			int i; // lcvs
			double mean, stddev;
			double mean1, mean2, stddev1, stddev2;
			double logl1 = 0, logl2 = 0;
			double mprob;

			mean = n.proj_sum[p];
			stddev = Math.sqrt(n.proj_sum_sq[p] - mean * mean);

			Stats stats = get_stats(n, threshold, p);
			stddev1 = stats.stddev1;
			stddev2 = stats.stddev2;
			mean1 = stats.mean1;
			mean2 = stats.mean2;

			if (stddev1 == 0 || stddev2 == 0) {
				return 1;
			}

			double edge1, edge2;
			for (i = 0; i < NUM_BINS_SMALL; i++) {
				if (Math.abs(i - (int) (NUM_BINS_SMALL / 2) + 0.5) > threshold) {
					// outside -- 2
					edge1 = mean + (-1 + ((double) (i)) * 2 / NUM_BINS_SMALL) * stddev;
					edge1 = (edge1 - mean2) / stddev2;
					edge2 = mean + (-1 + ((double) (i + 1)) * 2 / NUM_BINS_SMALL) * stddev;
					edge2 = (edge2 - mean2) / stddev2;
					mprob = (erf(edge2 / Math.sqrt(2)) - erf(edge1 / Math.sqrt(2))) / 2;
					if (mprob < 0)
						throw new RuntimeException("Error in prob calculation");

					logl2 += n.smallbin[p][i] * Math.log(mprob);
				} else {
					// inside -- 1
					edge1 = mean + (-1 + ((double) (i)) * 2 / NUM_BINS_SMALL) * stddev;
					edge1 = (edge1 - mean1) / stddev1;
					edge2 = mean + (-1 + ((double) (i + 1)) * 2 / NUM_BINS_SMALL) * stddev;
					edge2 = (edge2 - mean1) / stddev1;
					mprob = (erf(edge2 / Math.sqrt(2)) - erf(edge1 / Math.sqrt(2))) / 2;
					if (mprob < 0)
						throw new RuntimeException("Error in prob calculation");

					logl1 += n.smallbin[p][i] * Math.log(mprob);
				}
			}

			for (i = 0; i < NUM_BINS_LARGE; i++) {
				if (i >= (threshold - (int) (NUM_BINS_SMALL / 2) - 0.5)) {
					// outside the threshold

					// left side
					edge1 = mean - ((double) (i) + 2) * stddev;
					edge1 = (edge1 - mean2) / stddev2;
					edge2 = mean - ((double) (i) + 1) * stddev;
					edge2 = (edge2 - mean2) / stddev2;
					mprob = (erf(edge2 / Math.sqrt(2)) - erf(edge1 / Math.sqrt(2))) / 2;
					if (mprob < 0)
						throw new RuntimeException("Error in prob calculation");

					if (mprob > 0) {
						logl2 += n.largebin[p][2 * i] * Math.log(mprob);
					}

					// right side
					edge1 = mean + ((double) (i) + 1) * stddev;
					edge1 = (edge1 - mean2) / stddev2;
					edge2 = mean + ((double) (i) + 2) * stddev;
					edge2 = (edge2 - mean2) / stddev2;
					mprob = (erf(edge2 / Math.sqrt(2)) - erf(edge1 / Math.sqrt(2))) / 2;
					if (mprob < 0)
						throw new RuntimeException("Error in prob calculation");

					if (mprob > 0) {
						logl2 += n.largebin[p][2 * i + 1] * Math.log(mprob);
					}

				} else {
					// inside the threshold

					// left side
					edge1 = mean - ((double) (i) + 2) * stddev;
					edge1 = (edge1 - mean1) / stddev1;
					edge2 = mean - ((double) (i) + 1) * stddev;
					edge2 = (edge2 - mean1) / stddev1;
					mprob = (erf(edge2 / Math.sqrt(2)) - erf(edge1 / Math.sqrt(2))) / 2;
					if (mprob < 0)
						throw new RuntimeException("Error in prob calculation");

					if (mprob > 0) {
						logl1 += n.largebin[p][2 * i] * Math.log(mprob);
					}

					// right side
					edge1 = mean + ((double) (i) + 1) * stddev;
					edge1 = (edge1 - mean1) / stddev1;
					edge2 = mean + ((double) (i) + 2) * stddev;
					edge2 = (edge2 - mean1) / stddev1;
					mprob = (erf(edge2 / Math.sqrt(2)) - erf(edge1 / Math.sqrt(2))) / 2;
					if (mprob < 0)
						throw new RuntimeException("Error in prob calculation");

					if (mprob > 0) {
						logl1 += n.largebin[p][2 * i + 1] * Math.log(mprob);
					}
				}
			}
			return logl1 + logl2;
		}

		//
		// internal function to get the relavant statistics for parameter update
		//
		Stats get_stats(RpNode n, int threshold, int p) {

			Stats stats = new Stats();
			int i; // lcvs
			double sum1 = 0, sum2 = 0, sum_sq1 = 0, sum_sq2 = 0;
			double p1 = 0, p2 = 0;
			// 1 is for inside, 2 for outside.
			double mean, stddev, bin_mean;

			mean = n.proj_sum[p];
			stddev = Math.sqrt(n.proj_sum_sq[p] - mean * mean);

			// mean and stddev from the small bins
			for (i = 0; i < NUM_BINS_SMALL; i++) {
				bin_mean = mean - stddev + 2 * (i + 0.5) / NUM_BINS_SMALL * stddev;
				if (n.smallbin[p][i] > 0) {
					if (Math.abs(i - (int) (NUM_BINS_SMALL / 2) + 0.5) > threshold) {
						sum2 += bin_mean * n.smallbin[p][i];
						sum_sq2 += bin_mean * bin_mean * n.smallbin[p][i];
						p2 += n.smallbin[p][i];
					} else {
						sum1 += bin_mean * n.smallbin[p][i];
						sum_sq1 += bin_mean * bin_mean * n.smallbin[p][i];
						p1 += n.smallbin[p][i];
					}
				}
			}

			for (i = 0; i < NUM_BINS_LARGE; i++) {
				if (i >= (threshold - (int) (NUM_BINS_SMALL / 2) - 0.5)) {
					bin_mean = mean + ((int) (i / 2) + 1.5) * stddev * (-1);
					if (n.largebin[p][2 * i] > 0) {
						sum2 += bin_mean * n.largebin[p][2 * i];
						sum_sq2 += bin_mean * bin_mean * n.largebin[p][2 * i];
						p2 += n.largebin[p][2 * i];
					}
					bin_mean = mean + ((int) (i / 2) + 1.5) * stddev * 1;
					if (n.largebin[p][2 * i + 1] > 0) {
						sum2 += bin_mean * n.largebin[p][2 * i + 1];
						sum_sq2 += bin_mean * bin_mean * n.largebin[p][2 * i + 1];
						p2 += n.largebin[p][2 * i + 1];
					}
				} else {
					bin_mean = mean + ((int) (i / 2) + 1.5) * stddev * (-1);
					if (n.largebin[p][2 * i] > 0) {
						sum1 += bin_mean * n.largebin[p][2 * i];
						sum_sq1 += bin_mean * bin_mean * n.largebin[p][2 * i];
						p1 += n.largebin[p][2 * i];
					}
					bin_mean = mean + ((int) (i / 2) + 1.5) * stddev * 1;
					if (n.largebin[p][2 * i + 1] > 0) {
						sum1 += bin_mean * n.largebin[p][2 * i + 1];
						sum_sq1 += bin_mean * bin_mean * n.largebin[p][2 * i + 1];
						p1 += n.largebin[p][2 * i + 1];
					}
				}
			}

			stats.mean1 = sum1 / p1;
			stats.mean2 = sum2 / p2;
			if (sum_sq1 / p1 - sum1 / p1 * sum1 / p1 < -0.0001 * (sum_sq1 / p1))
				throw new RuntimeException("error in variance");
			if (sum_sq1 / p1 - sum1 / p1 * sum1 / p1 < 0) {
				stats.stddev1 = 0;
			} else {
				stats.stddev1 = Math.sqrt(sum_sq1 / p1 - sum1 / p1 * sum1 / p1);
			}

			if (sum_sq2 / p2 - sum2 / p2 * sum2 / p2 < -0.0001 * (sum_sq2 / p2))
				throw new RuntimeException("error in variance");
			if (sum_sq2 / p2 - sum2 / p2 * sum2 / p2 < 0) {
				stats.stddev2 = 0;
			} else {
				stats.stddev2 = Math.sqrt(sum_sq2 / p2 - sum2 / p2 * sum2 / p2);
			}

			return stats;
		}

		//
		// internal function to update the threshold value of a particular node
		// in the tree
		// Parameters: n - pointer to the RPNode
		// proj_num - projection index to be updated
		// Returns: the value which maximizes the avg distance between points
		// being split
		//
		private double update_thresh(RpNode n, int proj_num) {
			int i, j; // lcvs

			double thresh = 0;

			double mean = n.proj_sum[proj_num];
			double stddev = Math.sqrt(n.proj_sum_sq[proj_num] - mean * mean);
			double maxx = 0, bin_mean;
			double p1, p2, m1, m2, tmp;

			int p = proj_num;

			for (i = 0; i <= NUM_BINS_SMALL; i++) {
				tmp = 0;
				p1 = 0;
				p2 = 0;
				m1 = 0;
				m2 = 0;
				for (j = 0; j < 2 * NUM_BINS_LARGE; j++) {
					if (j % 2 == 0) {
						p1 += n.largebin[p][j];
						bin_mean = mean + ((int) (j / 2) + 1.5) * stddev * (-1);
						if (n.largebin[p][j] > 0)
							m1 += bin_mean * n.largebin[p][j];
					} else {
						p2 += n.largebin[p][j];
						bin_mean = mean + ((int) (j / 2) + 1.5) * stddev * 1;
						if (n.largebin[p][j] > 0)
							m2 += bin_mean * n.largebin[p][j];
					}
				}
				for (j = 0; j < NUM_BINS_SMALL; j++) {
					if (j < i) {
						p1 += n.smallbin[p][j];
						bin_mean = mean - stddev + 2 * (j + 0.5) / NUM_BINS_SMALL * stddev;
						if (n.smallbin[p][j] > 0)
							m1 += bin_mean * n.smallbin[p][j];
					} else {
						p2 += n.smallbin[p][j];
						bin_mean = mean - stddev + 2 * (j + 0.5) / NUM_BINS_SMALL * stddev;
						if (n.smallbin[p][j] > 0)
							m2 += bin_mean * n.smallbin[p][j];
					}
				}
				tmp = p1 * p2 * Math.pow(m1 / p1 - m2 / p2, 2);

				if (tmp > maxx) {
					maxx = tmp;
					thresh = mean - stddev + i * stddev * 2 / NUM_BINS_SMALL;
				}
			}
			n.threshold = thresh;
			return maxx;
		}

		//
		// internal function to update the count statistics of a particular node
		// in the RPTree
		// Parameters: n - pointer to RPNode to be updated
		// data - projected data which would be used to update the parameter
		// proj_num - projection index being used
		// offset - number of points that have passed through this node
		//
		private void update_hist(RpNode n, double data, int proj_num, int offset) {
			// histogram always sums to one, except at the begining
			int p = proj_num;
			int i, j; // lcvs
			double mean, stddev;
			double alpha;

			int count = n.total - offset + 1; // update starts after offset
			if (count > m_decay_count) {
				alpha = 1.0 / m_decay_count;
			} else {
				alpha = 1.0 / count;
			}

			mean = n.proj_sum[p]; // / sample_size;
			stddev = Math.sqrt(Math.abs(n.proj_sum_sq[p] - mean * mean));

			// reweigh all the bins
			for (i = 0; i < NUM_BINS_SMALL; i++) {
				n.smallbin[p][i] *= 1 - alpha;
			}
			for (j = 0; j < 2 * NUM_BINS_LARGE; j++) {
				n.largebin[p][j] *= 1 - alpha;
			}

			if (Math.abs(data - mean) < stddev || stddev == 0) { // within 1
																	// stddev |
																	// not
																	// enough
																	// data

				int ndx = 0;
				if (stddev != 0)
					ndx = (int) ((data - mean + stddev) * NUM_BINS_SMALL / (2 * stddev)); // index
				// in
				// which
				// projected
				// data
				// falls;
				if (ndx >= NUM_BINS_SMALL)
					ndx--; // correcting for rounding errors
				if (ndx < 0 || ndx >= NUM_BINS_SMALL) {
					throw new RuntimeException("Small bin index is too large " + ndx);
				}
				// update statistics for the corresponding bin
				n.smallbin[p][ndx] += alpha;
			} else { // more than 1 stddev away

				int tmp = ((int) (Math.abs(data - mean) / stddev)) - 1;
				int ndx = 2 * ((tmp < NUM_BINS_LARGE) ? tmp : (NUM_BINS_LARGE - 1)) + (((data - mean) < 0) ? 0 : 1);
				if (ndx < 0 || ndx >= 2 * NUM_BINS_LARGE) {
					throw new RuntimeException("Large bin index is too large " + ndx);
				}

				// update statistics for the corresponding bin
				n.largebin[p][ndx] += alpha;
			}

		}

		private void update_stats(RpNode n, double proj_value, int proj, double alpha) {
			n.proj_sum[proj] = n.proj_sum[proj] * (1 - alpha) + proj_value * alpha;
			n.proj_sum_sq[proj] = n.proj_sum_sq[proj] * (1 - alpha) + proj_value * proj_value * alpha;
		}

		//public ArrayList<Integer> invalidatedBuckets() {
		//	return m_invalidatedBuckets;
		//}

		public int getVectorLength() {
			return m_vect_length;
		}

	}
	
	

	@Override
	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		// TODO Auto-generated method stub
		super.prepareForUseImpl(monitor, repository);
		if (contextUsedOption.getChosenIndex() == 0)
			m_context = new Context(Context.Memory.MAIN_MEMORY, null);
		else if (contextUsedOption.getChosenIndex() == 1)
			m_context = new Context(Context.Memory.OPENCL_MEMORY, null);
		else if (contextUsedOption.getChosenIndex() == 2)
			m_context = new Context(Context.Memory.HSA_MEMORY, null);
		m_stats = new Stats();
	}

	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	class Entry implements Comparable<Entry> {
		public Entry(double d, int cls, InstanceData i) {
			distance = d;
			class_index = cls;
			next = i;
		}

		InstanceData next;
		double distance;
		int class_index;

		@Override
		public int compareTo(Entry o) {
			if (distance > o.distance)
				return 1;
			if (distance < o.distance)
				return -1;
			return 0;
		}
	}

	class Heap {
		private ArrayList<Entry> m_list;
		private int m_k;
		private int m_num_classes;
		private double[] m_query_projections;
		private double[] m_projections;

		public Heap(int k, int num_classes, double[] projections) {
			m_list = new ArrayList(k);
			m_k = k;
			m_num_classes = num_classes;
			m_query_projections = projections;
			m_projections = new double[projections.length];
		}

		public void add(Entry e) {
			if (m_list.size() < m_k)
				m_list.add(e);
			else {
				if (e.distance > m_list.get(m_k - 1).distance)
					return;
				int pos = Collections.binarySearch(m_list, e);
				if (pos < 0)
					pos = -pos - 1;
				m_list.add(pos, e);
				m_list.remove(m_k);
			}
		}
		
		

		public boolean shouldTraverse(RpNode node) {
			
			if (node.proj_type == 1) {
				double distance = Math.abs(m_projections[node.proj_to_use] - m_query_projections[node.proj_to_use]);
				double threshold_distance = Math.abs(node.threshold - m_query_projections[node.proj_to_use]);
				// a threshold split
				if (distance >= threshold_distance) // distance between points is more/equals than distance to the threshold
					return true;
			} else if (node.proj_type == 2) {
				double distance = Math.abs(m_query_projections[node.proj_to_use] - m_projections[node.proj_to_use]);
				double threshold_distance = Math.abs(node.threshold- node.mean[node.proj_to_use]);
				
				// a center split - distance between points exceeds radius.
				if ((distance >= threshold_distance) || 
						(Math.abs(m_projections[node.proj_to_use]) + distance > Math.abs(node.threshold))) 
					return true;
			}
			return false;
		}

		public double[] projections() {
			return m_query_projections;
		}

		public double[] distribution() {
			double[] distribution = new double[m_num_classes];
			for (Entry e : m_list)
				++distribution[e.class_index];
			return distribution;

		}

		public InstanceData last() {
			// TODO Auto-generated method stub
			return m_list.get(m_list.size() - 1).next;
		}

		public int getK() {
			return m_k;
		}
	}

	private void search(Instance inst, Heap h, int node_idx) {
    if (node_idx >= m_tree.m_nodes.size())
      return;
		RpNode node = null;
		node = m_tree.m_nodes.get(node_idx);
		if (node.is_leaf) {
			updateHeap(inst, h, node_idx, true);
			return;
		}

		// empty node at the last level
		if (TreeUtil.level(node_idx) == m_tree.m_depth-1) 
			return;
		
		int other_node = -1;
		if (node.proj_type == 1) {

			if (h.m_query_projections[node.proj_to_use] > node.threshold) {
				search(inst, h, TreeUtil.rightChild(node_idx));
				other_node = TreeUtil.leftChild(node_idx);
			} else {
				search(inst, h, TreeUtil.leftChild(node_idx)); // search left
				other_node = TreeUtil.rightChild(node_idx);
			}
		}

		if (node.proj_type == 2) {

			if (Math.abs(h.m_query_projections[node.proj_to_use] - node.proj_sum[node.proj_to_use]) < node.threshold) {
				search(inst, h, TreeUtil.rightChild(node_idx));
				other_node = TreeUtil.leftChild(node_idx);
			} else {
				search(inst, h, TreeUtil.leftChild(node_idx)); // search left
				other_node = TreeUtil.rightChild(node_idx);
			}
		}

		if (other_node < m_tree.m_nodes.size() && h.shouldTraverse(m_tree.m_nodes.get(other_node)))
			search(inst, h, other_node);

	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		if ( m_buckets == null )
			assignNodes();

		if (m_evaluation_instance == null)
			m_evaluation_instance = new DenseInstanceBuffer(DenseInstanceBuffer.Kind.FLOAT_BUFFER,m_context, 1, inst.numAttributes());
		m_evaluation_instance.begin(Buffer.WRITE);
		m_evaluation_instance.set(inst, 0);
		m_evaluation_instance.commit();
		++m_stats.num_predictions; 
		double[] projections = new double[m_num_projections];
		m_tree.computeProjections(inst, projections);
		Heap h = new Heap(kOption.getValue(), inst.numClasses(), projections);
		if (defeatistOption.isSet()) {
			int bucket = m_tree.findNode(inst, projections);
			updateHeap(inst, h, bucket, false);
		} else {
			search(inst, h, 0);
		}
		//System.out.println("Predicted with " + num_prediction_calcs + " out of "+ m_window.size());
		return h.distribution();
	}
	
	class Stats
	{
		long num_prediction_calcs=0;
		long num_predictions=0;
		long num_nodes_visited = 0;
	}
	
	private Stats m_stats;

	private FJLT[] m_fjlt;
	private DenseInstanceBuffer m_evaluation_instance;
	private DenseInstanceBuffer m_source_transform_buffer;
	private Buffer m_target_buffer;
	private Buffer m_gpu_projection_buffer;
	private Buffer m_projection_result;
	private Operations m_gpu_operations;
	private DenseInstanceBuffer m_instance_buffer;
	private HashMap<Integer, DenseInstanceBuffer> m_gpu_buckets;
	private Buffer m_min_range;
	private Buffer m_max_range;
	private Buffer m_distances;
	private Instances m_dataset;
	private Buffer m_attribute_types;
	private Buffer m_sort_indices;
	private CLogsVarKeyJava m_sort;
	private Operations m_operations;
	private DenseInstanceBuffer m_world;
	
	private void updateHeap(Instance inst, Heap h, int bucket, boolean updateProjection) {
		
		ArrayList<InstanceData> candidates = m_buckets.get(bucket);
		if (candidates == null)
			return;
		m_stats.num_prediction_calcs += candidates.size();
		m_stats.num_nodes_visited++;
		if (m_context.memoryType() != Context.MAIN_MEMORY)
		{
			DenseInstanceBuffer bucketBuffer =m_gpu_buckets.get(bucket);
			if (bucketBuffer == null)
			{
				return;
			}
					
			m_gpu_distance.squareDistanceFloat(m_dataset,
					m_evaluation_instance,
					bucketBuffer,
					m_min_range,
					m_max_range,
					m_attribute_types,
					m_distances);
			
			if (candidates.size() < h.getK())
			{
				float[] distances = BufHelper.rbf(m_distances);
				for (int i = 0; i < candidates.size() ; ++ i)
				{
					InstanceData data = candidates.get(i);
					Entry newEntry = new Entry( distances[i], (int)data.m_instance.classValue(), data );
					h.add(newEntry);
				}
			}
			else
			{
				m_operations.prepareOrderKey(m_sort_indices, candidates.size());
				m_sort.sortFixedBuffer(m_distances, m_sort_indices, candidates.size());
				int k = h.getK();
				float[] distances = new float[k];
				int[] indices = new int[k];
				m_distances.mapBuffer(Buffer.READ);
				m_distances.readArray(0, distances);
				m_distances.commitBuffer();
				
				m_sort_indices.mapBuffer(Buffer.READ);
				m_sort_indices.readArray(0, indices);
				m_sort_indices.commitBuffer();
				for (int i = 0; i < k ; ++ i)
				{
					InstanceData data = candidates.get(indices[i]);
					Entry newEntry = new Entry( distances[i], (int)data.m_instance.classValue(), data );
					h.add(newEntry);
				}
			}
			
		}
		else
		{
			for (InstanceData next : candidates) {
				double distance = m_euclidean_distance.distance(inst, next.m_instance);
				Entry newEntry = new Entry(distance, (int) next.m_instance.classValue(), next);
				h.add(newEntry);
			}
		}
		h.m_projections = h.last().m_projection;
	}

	@Override
	public void resetLearningImpl() {
		m_tree = null;
		m_window = null;
		m_buckets = null;
		
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// project
		if (m_tree == null) {
			m_dataset = inst.dataset();
			m_num_projections = numProjectionsOption.getValue();
			m_depth = maxTreeDepthOption.getValue();
			m_window = new ArrayDeque<InstanceData>(windowSizeOption.getValue());
			if (m_fjlt == null )
			{
				init_gpu_part(inst);
			}
			
			m_tree = new Tree(m_num_projections, inst.numAttributes(), m_depth, windowSizeOption.getValue());
			
			
			m_euclidean_distance = new EuclideanDistance(inst.dataset());
		}
		m_buckets = null;
		double[] projections = new double[m_num_projections];
		m_tree.computeProjections(inst, projections);
		double classValue = inst.classValue();
		inst.setClassValue(0);
		m_tree.trainOnInstance(inst, projections);
		inst.setClassValue(classValue);
		//if (m_window.size() == windowSizeOption.getValue())
		//	m_window.poll();
		m_window.add(new InstanceData(inst, projections));

	}

	private void init_gpu_part(Instance inst) {
    
		m_instance_buffer  = new DenseInstanceBuffer(m_context, 1, inst.numAttributes());
		m_gpu_operations = new Operations(m_context);
		m_projection_result = new Buffer(m_context, DirectMemory.DOUBLE_SIZE* m_num_projections);
		m_gpu_projection_buffer = new Buffer(m_context, DirectMemory.DOUBLE_SIZE * inst.numAttributes() * m_num_projections);

		int k_target = (int)Math.sqrt(inst.numAttributes());
		int transformers_needed = (int)(20/k_target);
		transformers_needed ++;
		m_fjlt = new FJLT[transformers_needed];
		for (int i = 0; i < m_fjlt.length; ++i )
			m_fjlt[i] = new FJLT(m_context, inst.numAttributes(), k_target);
		m_source_transform_buffer = new DenseInstanceBuffer(m_context, 1, inst.numAttributes());
		m_target_buffer = new Buffer(m_context, k_target * DirectMemory.DOUBLE_SIZE);
		m_world = new DenseInstanceBuffer(DenseInstanceBuffer.Kind.FLOAT_BUFFER, m_context, windowSizeOption.getValue(),  m_dataset.numAttributes());
		
		m_min_range = new Buffer(m_context, DirectMemory.FLOAT_SIZE *  m_dataset.numAttributes());
		m_max_range = new Buffer(m_context, DirectMemory.FLOAT_SIZE * m_dataset.numAttributes());
		m_distances = new Buffer(m_context, DirectMemory.FLOAT_SIZE * windowSizeOption.getValue());
		m_sort_indices  = new Buffer(m_context, DirectMemory.INT_SIZE * windowSizeOption.getValue());
		m_attribute_types= new Buffer(m_context, DirectMemory.INT_SIZE *  m_dataset.numAttributes());
		m_attribute_types.fill((byte)0);
		
		if (m_context.memoryType() != Context.MAIN_MEMORY)
		{
			m_gpu_distance = new Distance(m_context);
			m_min_max = new MinMax(m_context);
			m_sort = new CLogsVarKeyJava(m_context, true, "unsigned int", "unsigned int");
			m_operations = new Operations(m_context);
		}


	}

	protected void assignNodes() {
		
		m_gpu_buckets = new HashMap<Integer, DenseInstanceBuffer>();
		m_buckets = new HashMap<Integer, ArrayList<InstanceData>>();
		for (InstanceData i : m_window) {
			int node = m_tree.findNode(i.m_instance, i.m_projection);
			ArrayList<InstanceData> bucket = m_buckets.get(node);
			if (bucket == null) {
				bucket = new ArrayList<InstanceData>();
				bucket.add(i);
				m_buckets.put(node, bucket);
			} else {
				bucket.add(i);
			}
			
			if (m_context.memoryType() == Context.MAIN_MEMORY)
				m_euclidean_distance.update(i.m_instance);
		}
		if ((m_context.memoryType() != Context.MAIN_MEMORY))
		{
			m_world.begin(Buffer.WRITE);
			for (InstanceData i : m_window) 
				m_world.append(i.m_instance);
			m_world.commit();
			
			m_min_max.fullMinMaxFloat(m_dataset, m_world, m_min_range, m_max_range);			
			
			m_gpu_buckets = new HashMap<Integer, DenseInstanceBuffer>();
			Iterator<Integer> nodes = m_buckets.keySet().iterator();
			while(nodes.hasNext()) 
			{
				Integer next = nodes.next();
				ArrayList<InstanceData> data = m_buckets.get(next);
				
				DenseInstanceBuffer buf = new DenseInstanceBuffer(DenseInstanceBuffer.Kind.FLOAT_BUFFER, m_context, 
						data.size(), 
						m_dataset.numAttributes());
				buf.begin(Buffer.WRITE);
				for (InstanceData i : data)
					buf.append(i.m_instance);
				buf.commit();
				m_gpu_buckets.put(next, buf);
				
			}
			if (m_buckets.size() != m_gpu_buckets.size())
				throw new RuntimeException("The gpu buckets were not cleaned up properly");
		 }
		
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		double bucket_size = 0;
		Iterator<Integer> nodes = m_buckets.keySet().iterator();
		while(nodes.hasNext()) 
		{
			Integer next = nodes.next();
			ArrayList<InstanceData> data = m_buckets.get(next);
			bucket_size += data.size();
		}
		return new Measurement[]{
				new Measurement("Average number of calculations per prediction", ((double)m_stats.num_prediction_calcs)/m_stats.num_predictions  ),
				new Measurement("Average bucket size", (bucket_size)/m_buckets.size()  ),
				new Measurement("Average leaves visited", ((double)m_stats.num_nodes_visited)/m_stats.num_predictions  )
		};
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		out.append("Random Projection Tree");
	}

	public static double erf(double x) {
		// constants
		final double a1 = 0.254829592;
		final double a2 = -0.284496736;
		final double a3 = 1.421413741;
		final double a4 = -1.453152027;
		final double a5 = 1.061405429;
		final double p = 0.3275911;

		// Save the sign of x
		double sign = 1;
		if (x < 0) {
			sign = -1;
		}
		x = Math.abs(x);

		// A&S formula 7.1.26
		double t = 1.0 / (1.0 + p * x);
		double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

		return sign * y;
	}

}
