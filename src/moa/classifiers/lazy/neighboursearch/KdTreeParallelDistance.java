package moa.classifiers.lazy.neighboursearch;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.opencl.util.BufHelper;
import org.moa.opencl.util.CLogsSort;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.moa.opencl.util.Distance;
import org.moa.opencl.util.DoubleMergeSort;
import org.moa.opencl.util.MinMax;
import org.moa.opencl.util.Operations;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

import moa.classifiers.lazy.neighboursearch.KDTree;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch.MyHeap;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch.MyHeapElement;
import moa.classifiers.lazy.neighboursearch.kdtrees.KDTreeNode;
import weka.core.Instance;
import weka.core.Instances;

class Target {
	Instance m_target;
	DenseInstanceBuffer attributes;
}

/**
 * A simple illustration of the k-d tree query parallelisation.
 * 
 *
 */
public class KdTreeParallelDistance extends KDTree {

	private static final int OFFLOAD_THRESHOLD = 256;

	static {
		System.loadLibrary("viennacl-java-binding");
	}

	private transient DenseInstanceBuffer m_buffer;
	private transient Buffer m_indices;
	private transient Context m_context;
	private transient Target m_target;
	private transient Buffer m_min_range;
	private transient Buffer m_max_range;

	private transient Buffer m_min_range_split;
	private transient Buffer m_max_range_split;
	private transient Buffer m_indices_split;

	private transient Buffer m_instance_types;
	private transient Buffer m_distance_results;
	private transient Operations m_operations;
	// private DoubleMergeSort m_merge_sorter;
	private transient CLogsVarKeyJava m_merge_sorter;
	private transient Buffer m_sort_indices;
	private transient Distance m_distance;
	private transient MinMax m_min_max;

	private transient EuclideanDistance m_override = null;

	public KdTreeParallelDistance(Context ctx, DenseInstanceBuffer buffer, Buffer indices) {
		super();
		m_context = ctx;
		m_buffer = buffer;
		m_indices = indices;
		// m_merge_sorter = new DoubleMergeSort(m_context, m_buffer.rows());
		m_merge_sorter = new CLogsVarKeyJava(ctx, true, "unsigned int", "unsigned int");
		m_sort_indices = new Buffer(m_context, m_buffer.rows() * DirectMemory.INT_SIZE);
		m_distance_results = new Buffer(m_context, m_buffer.rows() * DirectMemory.FLOAT_SIZE);

		m_target = new Target();
		m_MaxInstInLeaf = OFFLOAD_THRESHOLD;
		m_operations = new Operations(m_context);
		m_MeasurePerformance = true;
		m_distance = new Distance(m_context);
		m_min_max = new MinMax(ctx);

		m_min_range_split = new Buffer(m_context, buffer.attributes().byteSize() / buffer.rows());
		m_max_range_split = new Buffer(m_context, buffer.attributes().byteSize() / buffer.rows());

		m_indices_split = new Buffer(m_context, DirectMemory.INT_SIZE * buffer.rows());
		m_override = new EuclideanDistance() {

			public double[][] initializeRanges(int[] instList, int startIdx, int endIdx) throws Exception {
				if (m_Data == null)
					throw new Exception("No instances supplied.");
				int numAtt = m_Data.numAttributes();
				// int bufAtt =
				// (int)m_buffer.attributes().byteSize()/(m_buffer.rows() *
				// DirectMemory.FLOAT_SIZE);
				double[][] ranges = new double[numAtt][3];
				double[][] spot = new double[numAtt][3];
				int length = endIdx + 1 - startIdx; // inclusive range
				int[] range = new int[length];
				System.arraycopy(instList, startIdx, range, 0, length);
				m_indices_split.mapBuffer(Buffer.WRITE);
				m_indices_split.writeArray(0, range);
				m_indices_split.commitBuffer();
				// m_buffer.write(m_Data);
				m_min_max.fullMinMaxFloatIndices(m_Data, m_buffer, m_min_range_split, m_max_range_split,
						m_indices_split, length);

				float[] min_values = BufHelper.rbf(m_min_range_split);
				float[] max_values = BufHelper.rbf(m_max_range_split);

				for (int i = 0; i < numAtt; ++i) {
					spot[i][R_MIN] = min_values[i];
					spot[i][R_MAX] = max_values[i];
					spot[i][R_WIDTH] = Math.abs(max_values[i] - min_values[i]);
				}

				/*
				 * if (m_Data.numInstances() <= 0) {
				 * initializeRangesEmpty(numAtt, ranges); return ranges; } else
				 * { // initialize ranges using the first instance
				 * updateRangesFirst(m_Data.instance(instList[startIdx]),
				 * numAtt, ranges); // update ranges, starting from the second
				 * for (int i = startIdx+1; i <= endIdx; i++) {
				 * updateRanges(m_Data.instance(instList[i]), numAtt, ranges); }
				 * } for (int i = 0; i < numAtt; ++i) { if (Math.abs(spot[i][0]
				 * - ranges[i][0]) > 0.01 && m_Data.classIndex() != i) {
				 * dumpWindow(m_Data, startIdx, endIdx, instList, i ,
				 * spot[i][1], ranges[i][1]); System.out.println(); }
				 * 
				 * if (Math.abs(spot[i][1] - ranges[i][1]) > 0.01 &&
				 * m_Data.classIndex() != i ) { dumpWindow(m_Data, startIdx,
				 * endIdx, instList, i , spot[i][1], ranges[i][1]);
				 * m_min_max.fullMinMaxFloat(m_Data, m_buffer,
				 * m_min_range_split, m_max_range_split); float[] max =
				 * BufHelper.rbf(m_max_range_split); System.out.println(); } }
				 */
				return ranges;
			}

			void dumpWindow(Instances data, int startIdx, int endIdx, int[] instList, int attIdx, double bad,
					double good) {
				for (int i = 0; i < data.numInstances(); ++i) {
					// for (int att = startIdx; att <= endIdx); ++att)
					// if (Math.abs (data.instance(i).value(att) -bad) < 0.001)
					// System.out.println("bad "+ i + " expected att "+ attIdx +
					// " but was "+ att);
					if (Math.abs(data.instance(instList[i]).value(attIdx) - good) < 0.0001)
						System.out.println("good " + i);

				}
			}

			public double[][] initializeRanges() {
				if (m_Data == null) {
					m_Ranges = null;
					return m_Ranges;
				}
				int numAtt = m_Data.numAttributes();
				double[][] ranges = new double[numAtt][3];
				m_min_max.fullMinMaxFloat(m_Data, m_buffer, m_min_range_split, m_max_range_split);

				float[] min_values = BufHelper.rbf(m_min_range_split);
				float[] max_values = BufHelper.rbf(m_max_range_split);

				for (int i = 0; i < numAtt; ++i) {
					ranges[i][R_MIN] = min_values[i];
					ranges[i][R_MAX] = max_values[i];
					ranges[i][R_WIDTH] = Math.abs(max_values[i] - min_values[i]);
				}
				/*
				 * if (m_Data.numInstances() <= 0) {
				 * initializeRangesEmpty(numAtt, ranges); m_Ranges = ranges;
				 * return m_Ranges; } else { // initialize ranges using the
				 * first instance updateRangesFirst(m_Data.instance(0), numAtt,
				 * ranges); }
				 * 
				 * // update ranges, starting from the second for (int i = 1; i
				 * < m_Data.numInstances(); i++)
				 * updateRanges(m_Data.instance(i), numAtt, ranges);
				 */
				m_Ranges = ranges;

				return m_Ranges;
			}
		};
	}

	@Override
	protected void buildKDTree(Instances instances) throws Exception {
		checkMissing(instances);
		m_buffer.write(m_Instances);

		m_DistanceFunction = m_EuclideanDistance = m_override;
		m_EuclideanDistance.setInstances(instances);

		m_Instances = instances;
		int numInst = m_Instances.numInstances();

		// Make the global index list
		m_InstList = new int[numInst];

		for (int i = 0; i < numInst; i++) {
			m_InstList[i] = i;
		}

		double[][] universe = m_EuclideanDistance.getRanges();

		// initializing internal fields of KDTreeSplitter
		m_Splitter.setInstances(m_Instances);
		m_Splitter.setInstanceList(m_InstList);
		m_Splitter.setEuclideanDistanceFunction(m_EuclideanDistance);
		m_Splitter.setNodeWidthNormalization(m_NormalizeNodeWidth);

		// building tree
		m_NumNodes = m_NumLeaves = 1;
		m_MaxDepth = 0;
		m_Root = new KDTreeNode(m_NumNodes, 0, m_Instances.numInstances() - 1, universe);

		splitNodes(m_Root, universe, m_MaxDepth + 1);
		refreshGPUCache();
	}

	private void refreshGPUCache() throws Exception {

		if (m_min_range == null) {
			
			m_min_range = new Buffer(m_context, m_buffer.attributes().byteSize() / m_buffer.rows());
			m_max_range = new Buffer(m_context, m_buffer.attributes().byteSize() / m_buffer.rows());
			
			m_instance_types = new Buffer(m_context, DirectMemory.INT_SIZE * m_Instances.numAttributes());
			m_instance_types.mapBuffer(Buffer.WRITE);
			m_instance_types.writeArray(0, attributeTypes(m_Instances));
			m_instance_types.commitBuffer();
		}

	//	m_buffer.write(m_Instances);
		m_indices.mapBuffer(Buffer.WRITE);
		m_indices.writeArray(0, m_InstList);
		m_indices.commitBuffer();
		
		m_min_max.fullMinMaxFloat(m_Instances, m_buffer, m_min_range, m_max_range);
	}

	@Override
	public void update(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		super.update(instance);
		refreshGPUCache();
	}

	protected void findNearestNeighbours(Instance target, KDTreeNode node, int k, MyHeap heap, double distanceToParents)
			throws Exception {
		if (node.isALeaf()) {
			if (node.m_End - node.m_Start < OFFLOAD_THRESHOLD) {
				super.findNearestNeighbours(target, node, k, heap, distanceToParents);
				return;
			}
			findDistances(node.m_Start, node.m_End, target, 0);
			
			float[] check = BufHelper.rbf(m_distance_results);
			int read_len = Math.min(k, node.m_End - node.m_Start + 1);
			float[] dist = new float[read_len];
			int[] ind = new int[read_len];

			m_operations.prepareOrderKey(m_sort_indices, node.m_End - node.m_Start + 1);
			m_merge_sorter.sortFixedBuffer(m_distance_results, m_sort_indices, node.m_End - node.m_Start + 1);

			readSortIndices(ind);
			readDistances(dist);

			for (int i = 0; i < Math.min(k, node.m_End - node.m_Start + 1); ++i) {
				double distance = dist[i];
				int instIndex = m_InstList[node.m_Start + ind[i]];
				if (target == m_Instances.instance(instIndex)) // for
					// hold-one-out
					// cross-validation
					continue;

				if (heap.size() < k) {
					heap.put(instIndex, distance);
				} else {
					MyHeapElement temp = heap.peek();
					if (distance < temp.distance) {
						heap.putBySubstitute(instIndex, distance);
					} else if (distance == temp.distance) {
						heap.putKthNearest(instIndex, distance);
					} else
						return; // no more suitable candidates
				} // end else heap.size==k

			}

		} else {
			KDTreeNode nearer, further;
			boolean targetInLeft = m_EuclideanDistance.valueIsSmallerEqual(target, node.m_SplitDim, node.m_SplitValue);

			if (targetInLeft) {
				nearer = node.m_Left;
				further = node.m_Right;
			} else {
				nearer = node.m_Right;
				further = node.m_Left;
			}
			findNearestNeighbours(target, nearer, k, heap, distanceToParents);

			// ... now look in further half if maxDist reaches into it
			if (heap.size() < k) { // if haven't found the first k
				double distanceToSplitPlane = distanceToParents + m_EuclideanDistance.sqDifference(node.m_SplitDim,
						target.value(node.m_SplitDim), node.m_SplitValue);
				findNearestNeighbours(target, further, k, heap, distanceToSplitPlane);
				return;
			} else { // else see if ball centered at query intersects with the
						// other
						// side.
				double distanceToSplitPlane = distanceToParents + m_EuclideanDistance.sqDifference(node.m_SplitDim,
						target.value(node.m_SplitDim), node.m_SplitValue);
				if (heap.peek().distance >= distanceToSplitPlane) {
					findNearestNeighbours(target, further, k, heap, distanceToSplitPlane);
				}
			} // end else
		} // end else_if an internal node
	}

	private void readDistances(float[] dist) {
		m_distance_results.mapBuffer(Buffer.READ, 0, dist.length * DirectMemory.FLOAT_SIZE);
		m_distance_results.readArray(0, dist);
		m_distance_results.commitBuffer();
	}

	private void readSortIndices(int[] ind) {
		m_sort_indices.mapBuffer(Buffer.READ, 0, ind.length * DirectMemory.INT_SIZE);
		m_sort_indices.readArray(0, ind);
		m_sort_indices.commitBuffer();
	}

	private void findDistances(int start, int end, Instance target, double max_length) {

		if (m_target.m_target != target) {
			m_target.m_target = target;
			m_target.attributes = new DenseInstanceBuffer(DenseInstanceBuffer.Kind.FLOAT_BUFFER, m_context, 1,
					target.numAttributes());
			m_target.attributes.begin(Buffer.WRITE);
			m_target.attributes.set(target, 0);
			m_target.attributes.commit();
		}
		m_distance.squareDistanceFloat(target.dataset(), m_target.attributes, m_buffer, m_min_range, m_max_range,
				m_instance_types, m_distance_results,  end - start +1, m_indices, start);
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

	private void createProgram(Context ctx) {
		StringBuffer src = loadKernel("distance.cl");
		m_context.add("kd_tree_distance_helper", src.toString());
	}

	protected StringBuffer loadKernel(String name) {
		String line;
		InputStream is = getClass().getResourceAsStream(name);
		StringBuffer data = new StringBuffer();

		BufferedReader r = new BufferedReader(new InputStreamReader(is));
		try {
			while ((line = r.readLine()) != null)
				data.append(line).append('\n');
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return data;
	}

	private int[] attributeTypes(Instances dataset) {
		int[] attributeTypes = new int[dataset.numAttributes()];
		for (int i = 0; i < attributeTypes.length; ++i) {
			if (dataset.attribute(i).isNumeric())
				attributeTypes[i] = 0;
			else if (dataset.attribute(i).isNominal())
				attributeTypes[i] = 1;
			else
				attributeTypes[i] = 2;
		}
		return attributeTypes;
	}

}
