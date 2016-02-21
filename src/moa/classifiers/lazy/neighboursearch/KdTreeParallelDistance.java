package moa.classifiers.lazy.neighboursearch;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import org.moa.gpu.DenseInstanceBuffer;
import org.moa.opencl.util.CLogsSort;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.moa.opencl.util.Distance;
import org.moa.opencl.util.DoubleMergeSort;
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

/**
 * A simple illustration of the k-d tree query parallelisation.
 * 
 * @author john
 *
 */
public class KdTreeParallelDistance extends KDTree {

	private static final int OFFLOAD_THRESHOLD = 1024;

	static {
		System.loadLibrary("viennacl-java-binding");
	}

	private DenseInstanceBuffer m_buffer;
	private Buffer m_indices;
	private Context m_context;
	private Target m_target;
	private Buffer m_min_range;
	private Buffer m_width;
	private Buffer m_instance_types;
	private Buffer m_distance_results;
	private Operations m_operations;
	//private DoubleMergeSort m_merge_sorter;
	private CLogsVarKeyJava m_merge_sorter;
	private Buffer m_sort_indices;
	private Distance m_distance;

	private class Target {
		Instance m_target;
		DenseInstanceBuffer attributes;
	}

	public KdTreeParallelDistance(Context ctx, DenseInstanceBuffer buffer, Buffer indices) {
		super();
		m_context = ctx;
		m_buffer = buffer;
		m_indices = indices;
		//m_merge_sorter = new DoubleMergeSort(m_context, m_buffer.rows());
		m_merge_sorter = new CLogsVarKeyJava(ctx, true, "unsigned int", "unsigned int");
		m_sort_indices = new Buffer(m_context, m_buffer.rows() * DirectMemory.INT_SIZE);
		m_distance_results = new Buffer(m_context, m_buffer.rows() * DirectMemory.FLOAT_SIZE);

		m_target = new Target();
		m_MaxInstInLeaf = 4 * OFFLOAD_THRESHOLD;
		m_operations = new Operations(m_context);
		m_MeasurePerformance  = true;
		m_distance = new Distance(m_context);
	}

	@Override
	protected void buildKDTree(Instances instances) throws Exception {
		// TODO Auto-generated method stub
		super.buildKDTree(instances);
		refreshGPUCache();
	}

	private void refreshGPUCache() throws Exception {

		if (m_min_range == null) {
			m_min_range = new Buffer(m_context, DirectMemory.FLOAT_SIZE * m_EuclideanDistance.getRanges().length);
			m_width = new Buffer(m_context, DirectMemory.FLOAT_SIZE * m_EuclideanDistance.getRanges().length);

			m_instance_types = new Buffer(m_context, DirectMemory.INT_SIZE * m_Instances.numAttributes());
			m_instance_types.mapBuffer(Buffer.WRITE);
			m_instance_types.writeArray(0, attributeTypes(m_Instances));
			m_instance_types.commitBuffer();
		}

		m_buffer.write(m_Instances);
		m_indices.mapBuffer(Buffer.WRITE);
		m_indices.writeArray(0, m_InstList);
		m_indices.commitBuffer();

		m_min_range.mapBuffer(Buffer.WRITE);
		m_width.mapBuffer(Buffer.WRITE);

		for (int i = 0; i < m_EuclideanDistance.getRanges().length; ++i) {
			m_min_range.write(i * DirectMemory.FLOAT_SIZE,
					m_EuclideanDistance.getRanges()[i][NormalizableDistance.R_MIN]);
			m_width.write(i * DirectMemory.FLOAT_SIZE,
					m_EuclideanDistance.getRanges()[i][NormalizableDistance.R_WIDTH]);
		}
		m_min_range.commitBuffer();
		m_width.commitBuffer();

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
			int read_len = Math.min(k, node.m_End - node.m_Start + 1);
			double[] dist = new double[read_len];
			int[] ind = new int[read_len];

			m_operations.prepareOrderKey(m_sort_indices, node.m_End - node.m_Start + 1);
			m_merge_sorter.sortFixedBuffer(m_distance_results, m_sort_indices, node.m_End - node.m_Start + 1);

			
			readSortIndices(ind);
			readDistances(dist);

			for (int i = 0; i < Math.min(k, node.m_End - node.m_Start + 1); ++i) {
				double distance = dist[i];
				int instIndex = m_InstList[ind[i]];
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

	private void readDistances(double[] dist) {
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
			m_target.attributes = new DenseInstanceBuffer(DenseInstanceBuffer.Kind.FLOAT_BUFFER, m_context, 1, target.numAttributes());
			m_target.attributes.begin(Buffer.WRITE);
			m_target.attributes.set(target, 0);
			m_target.attributes.commit();
		}
		/*
		m_distance_kernel.set_global_size(0, (int) (end - start + 1));
		m_distance_kernel.set_arg(0, m_indices);
		m_distance_kernel.set_arg(1, m_target.attributes.attributes());
		m_distance_kernel.set_arg(2, m_buffer.attributes());

		m_distance_kernel.set_arg(3, m_min_range);
		m_distance_kernel.set_arg(4, m_width);
		m_distance_kernel.set_arg(5, m_instance_types);
		m_distance_kernel.set_arg(6, m_distance_results);
		m_distance_kernel.set_arg(7, target.numAttributes());
		m_distance_kernel.set_arg(8, start);
		m_distance_kernel.invoke();
		*/
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
