package moa.classifiers.gpu;

import java.util.ArrayList;
import java.util.Random;

import org.moa.opencl.knn.DoubleLinearSearch;
import org.moa.opencl.knn.SimpleZOrderSearch;
import org.viennacl.binding.Context;
import org.viennacl.binding.GlobalContext;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.MultiChoiceOption;
import moa.tasks.TaskMonitor;
import weka.core.Instance;

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

	public MultiChoiceOption contextUsedOption = new MultiChoiceOption("contextUsed", 'c', "Context Type",
			new String[] { "CPU", "OPENCL", "HSA" }, new String[] { "CPU single thread",
					"OpenCL offload. Use OPENCL_DEVICE Env. variable to select device", "HSA Offload" },
			0);

	private Context m_context;

	private Tree m_tree;

	private int m_num_projections;

	private int m_depth;

	private class RpNode {
		public RpNode(int node_index, boolean leaf, int vect_length, int num_projections) {
			this.node_ndx = node_index;
			this.is_leaf = is_leaf;
			this.total = 0;
			this.mean = new double[vect_length];
			this.proj_sum = new double[num_projections];
			this.proj_sum_sq = new double[num_projections];
			this.smallbin = new double[num_projections][Tree.NUM_BINS_SMALL];
			this.largebin = new double[num_projections][2*Tree.NUM_BINS_LARGE];
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
		static final int DECAY_COUNT = 65536;
		static final int N1 = 50;
		static final int N2 = 50;
		static final int N3 = 250;
		private int m_num_projections;
		private int m_vect_length;
		private ArrayList<double[]> m_projections;
		private ArrayList<RpNode> m_nodes;
		private int m_depth;

		public Tree(int num_projections, int vect_length, int depth) {
			Random rnd = new Random(System.currentTimeMillis());
			m_depth = depth;
			m_projections = new ArrayList<double[]>();
			m_nodes = new ArrayList<RpNode>();
			m_num_projections = num_projections;
			m_vect_length = vect_length;
			for (int i = 0; i < m_num_projections; ++i) {
				double[] proj = new double[m_vect_length];
				for (int j = 0; j < m_vect_length; ++j) {
					proj[j] = rnd.nextDouble();
				}
				m_projections.add(proj);
			}
			for (int i = 0; i < (0x1 << m_depth) - 1; i++) {
				RpNode node = new RpNode(i, true, vect_length, num_projections);
				m_nodes.add(node);
			}
		}

		public void trainOnInstance(Instance inst) {
			// prepare projections for each of the splits (complexity
			// Nprojections x d^2)
			double[] projections = new double[m_num_projections];
			for (int i = 0; i < projections.length; ++i) {
				projections[i] = 0;
				for (int j = 0; j < inst.numAttributes(); ++j) {
					projections[i] += inst.isMissing(j) ? 0 : m_projections.get(i)[j] * inst.value(j);
				}
			}
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

		private void learnRpNode(RpNode n, Instance inst, double[] projections) {
			double alpha; // decay parameter
			int i, p; // counter

			n.total++; // count update

			if (n.total > DECAY_COUNT) {
				alpha = 1. / DECAY_COUNT;
			} else {
				alpha = 1. / (n.total);
			}

			for (i = 0; i < inst.numAttributes(); i++) { // mean update
				n.mean[i] = inst.isMissing(i) ? (1 - alpha) * (n.mean[i])
						: (1 - alpha) * (n.mean[i]) + inst.value(i) * alpha;
			}

			for (p = 0; p < m_num_projections; p++) { // proj sum / sum_sq
														// update
				update_stats(n, projections[p], p, alpha);
				if (n.total > N1) { // histogram update
					update_hist(n, projections[p], p, N1);
				}
			}

			if (n.total == N1 + N2) { // sufficient points to have a stable
										// threshold
				double maxx = 0, tmp, thresh =0;
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

	private void center_split(RpNode n) {
			// TODO Auto-generated method stub
			
		}

	// 
   // internal function to update the threshold value of a particular node in the tree
   // Parameters:    n    -  pointer to the RPNode
   //                proj_num  - projection index to be updated 
   // Returns:  the value which maximizes the avg distance between points being split
   //
	 private double update_thresh(RpNode n, int proj_num) {
	       int i,j; // lcvs
	       
	       double thresh = 0;
	     
	       double mean = n.proj_sum[proj_num]; 
	       double stddev = Math.sqrt( n.proj_sum_sq[proj_num]  - mean * mean);
	       double maxx = 0, bin_mean;
	       double p1,p2,m1,m2,tmp;
	     
	       int p = proj_num;
	     
	       for(i=0;i<=NUM_BINS_SMALL;i++) {
	         tmp = 0; p1=0; p2=0; m1=0; m2=0;
	         for(j=0;j<2*NUM_BINS_LARGE;j++) {
	           if(j%2==0) { 
	              p1+=n.largebin[p][j]; 
	              bin_mean = mean+((int)(j/2)+1.5)*stddev*(-1);
	              if(n.largebin[p][j]>0)
	                m1+= bin_mean * n.largebin[p][j];
	           }
	           else { 
	              p2+=n.largebin[p][j]; 
	              bin_mean = mean+((int)(j/2)+1.5)*stddev*1;
	              if(n.largebin[p][j]>0)
	                m2+=bin_mean * n.largebin[p][j]; 
	           }
	         }
	         for(j=0;j<NUM_BINS_SMALL;j++) {
	           if(j<i) { 
	              p1+=n.smallbin[p][j]; 
	              bin_mean = mean-stddev + 2*(j+0.5)/NUM_BINS_SMALL*stddev;
	              if(n.smallbin[p][j]>0)
	                m1 += bin_mean*n.smallbin[p][j]; 
	           }
	           else    { 
	              p2+=n.smallbin[p][j]; 
	              bin_mean = mean-stddev + 2*(j+0.5)/NUM_BINS_SMALL*stddev;
	              if(n.smallbin[p][j]>0)
	                m2 += bin_mean*n.smallbin[p][j]; 
	           }
	         }
	         tmp = p1*p2*Math.pow(m1/p1-m2/p2,2);
	     
	         if(tmp > maxx) {
	            maxx=tmp; 
	            thresh = mean - stddev + i*stddev*2/NUM_BINS_SMALL; 
	         }
	       }
	       n.threshold = thresh;
	       return maxx;
		}

		//
       // internal function to update the count statistics of a particular node in the RPTree
       // Parameters:     n     -  pointer to RPNode to be updated
       //                 data  -  projected data which would be used to update the parameter
       //                 proj_num  -  projection index being used
       //                 offset    -  number of points that have passed through this node
       //
		private void update_hist(RpNode n, double data, int proj_num, int offset) {
		      //histogram always sums to one, except at the begining 
		       int p = proj_num;
		       int i,j; //lcvs
		       double mean, stddev;
		       double alpha;
		     
		       int count = n.total - offset +1;   // update starts after offset
		       if(count>DECAY_COUNT){ alpha = 1.0/DECAY_COUNT;}
		       else{ alpha = 1.0/count; }
		     
		       mean = n.proj_sum[p]; // / sample_size;
		       stddev = Math.sqrt( n.proj_sum_sq[p]  - mean*mean);
		     
		       //reweigh all the bins
		       for(i=0;i<NUM_BINS_SMALL;i++){
		         n.smallbin[p][i] *= 1-alpha;
		       }
		       for(j=0;j<2*NUM_BINS_LARGE;j++){
		         n.largebin[p][j] *= 1-alpha;
		       }
		     
		       if( Math.abs(data-mean) < stddev )   { // within 1 stddev
		     
		         int ndx = (int) ((data-mean+stddev)*NUM_BINS_SMALL/(2*stddev)); // index in which projected data falls;
		         if(ndx>=NUM_BINS_SMALL) ndx--;  // correcting for rounding errors
		         if(ndx<0 || ndx>=NUM_BINS_SMALL) { throw new RuntimeException("Small bin index is too large "+ ndx); }
		            // update statistics for the corresponding bin
		         n.smallbin[p][ndx]+=alpha;
		       }
		       else  {  // more than 1 stddev away
		     
		         int tmp = ((int)( Math.abs(data-mean)/stddev))-1; 
		         int ndx = 2*((tmp<NUM_BINS_LARGE)? tmp : (NUM_BINS_LARGE-1)) + (((data-mean)<0)? 0:1);  
		         if(ndx<0 || ndx>=2*NUM_BINS_LARGE) { throw new RuntimeException("Large bin index is too large " + ndx);}
		     
		            // update statistics for the corresponding bin
		         n.largebin[p][ndx] += alpha;
		       }
			
		}

		private void update_stats(RpNode n, double proj_value, int proj, double alpha) {
	       n.proj_sum[proj] = n.proj_sum[proj]*(1-alpha) + proj_value*alpha;
	       n.proj_sum_sq[proj] = n.proj_sum_sq[proj]*(1-alpha) + proj_value*proj_value*alpha;
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
		else if (contextUsedOption.getChosenIndex() == 3)
			m_context = new Context(Context.Memory.HSA_MEMORY, null);
	}

	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void resetLearningImpl() {
		m_tree = null;

	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// project
		if (m_tree == null)
		{
			m_num_projections = 20;
			m_depth = 8;
			m_tree = new Tree(m_num_projections, inst.numAttributes(), m_depth);
		}
		m_tree.trainOnInstance(inst);
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		out.append("Random Projection Tree");
	}

}
