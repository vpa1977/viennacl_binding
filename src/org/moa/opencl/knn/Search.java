package org.moa.opencl.knn;


import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.SlidingWindow;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import moa.options.AbstractOptionHandler;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public abstract class Search  extends AbstractOptionHandler{
	
    /** no weighting. */
    public static final int WEIGHT_NONE = 1;
    /** weight by 1/distance. */
    public static final int WEIGHT_INVERSE = 2;
    /** weight by 1-distance. */
    public static final int WEIGHT_SIMILARITY = 4;

	private SlidingWindow m_sliding_window;
	private Buffer m_bounds;
	private int m_distance_weighting = WEIGHT_NONE;
	protected boolean m_dirty;

	public void setSlidingWindow(SlidingWindow sliding_window) {
		m_sliding_window = sliding_window;
	}
	
	public void setDistanceWeighting(int mode)
	{
		m_distance_weighting  = mode;
	}
	
	   /**
     * Turn the list of nearest neighbors into a probability distribution.
     *
     * @param neighbours the list of nearest neighboring instances
     * @param distances the distances of the neighbors
     * @return the probability distribution
     * @throws Exception if computation goes wrong or has no class attribute
     */
    
    protected double [] makeDistribution(int[] indices, Buffer distance_buffer, int k )
      throws Exception {

     int numClasses = m_sliding_window.dataset().numClasses();
      double total = 0, weight;
      double [] distribution = new double [numClasses];
      double[] distances = null;
      
      if (m_distance_weighting != WEIGHT_NONE)
      {
    	 distances = new double[indices.length];
    	 distance_buffer.mapBuffer(Buffer.READ, 0,indices.length * DirectMemory.DOUBLE_SIZE );
    	 distance_buffer.readArray(0, distances);
    	 distance_buffer.commitBuffer();
      }
      
      // Set up a correction to the estimator
      if (m_sliding_window.dataset().classAttribute().type() == Attribute.NOMINAL) {
        for(int i = 0; i < numClasses; i++) {
        	distribution[i] = 1.0 / Math.max(1,m_sliding_window.model().rows());
        }
        total = (double)numClasses / Math.max(1,m_sliding_window.model().rows());
      }

      for(int i=0; i < k; i++) {
        // Collect class counts
        switch (m_distance_weighting) {
          case WEIGHT_INVERSE:
            weight = 1.0 / (Math.sqrt(distances[i]/m_sliding_window.dataset().numAttributes()) + 0.001); // to avoid div by zero
            break;
          case WEIGHT_SIMILARITY:
            weight = 1.0 - Math.sqrt(distances[i]/m_sliding_window.dataset().numAttributes());
            break;
          default:                                 // WEIGHT_NONE:
            weight = 1.0;
            break;
        }
        weight *= m_sliding_window.weight(indices[i]);
        try {
          switch (m_sliding_window.dataset().classAttribute().type()) {
            case Attribute.NOMINAL:
              distribution[(int)m_sliding_window.classValue(indices[i])] += weight;
              break;
            case Attribute.NUMERIC:
              distribution[0] += m_sliding_window.classValue(indices[i]) * weight;
              break;
          }
        } catch (Exception ex) {
          throw new Error("Data has no class attribute!");
        }
        total += weight;      
      }

      // Normalise distribution
      if (total > 0) {
        Utils.normalize(distribution, total);
      }
      return distribution;
    }

	public abstract void init(Context m_context, Instances dataset);
	public abstract double[] getVotesForInstance(Instance instance, DenseInstanceBuffer data, int K ) throws Exception;

	public void markDirty() {
		m_dirty = true;
	}

}
