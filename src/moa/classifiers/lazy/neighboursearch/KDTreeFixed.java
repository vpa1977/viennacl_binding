package moa.classifiers.lazy.neighboursearch;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class KDTreeFixed extends KDTree {

	private EuclideanDistance m_override;

	public KDTreeFixed() {
		super();
                m_MaxInstInLeaf = 256;
		m_override = new EuclideanDistance() {
			public double[][] updateRanges(Instance instance, double[][] ranges) {
				// updateRangesFirst must have been called on ranges
				if (instance instanceof SparseInstance) {
					class SparseInstanceAccess extends SparseInstance {
						public SparseInstanceAccess(Instance i) {
							super(i);
						}

						public int[] getIndices() {
							return m_Indices;
						}
					}
					;

					SparseInstanceAccess access = new SparseInstanceAccess(instance);
					for (int j : access.getIndices()) {
						if (!instance.isMissing(j)) {
							double value = instance.value(j);
							if (value < ranges[j][R_MIN]) {
								ranges[j][R_MIN] = value;
								ranges[j][R_WIDTH] = ranges[j][R_MAX] - ranges[j][R_MIN];
							} else {
								if (instance.value(j) > ranges[j][R_MAX]) {
									ranges[j][R_MAX] = value;
									ranges[j][R_WIDTH] = ranges[j][R_MAX] - ranges[j][R_MIN];
								}
							}
						}

					}
				} else if (instance instanceof DenseInstance) {
					for (int j = 0; j < ranges.length; j++) {

						if (!instance.isMissing(j)) {
							double value = instance.value(j);
							if (value < ranges[j][R_MIN]) {
								ranges[j][R_MIN] = value;
								ranges[j][R_WIDTH] = ranges[j][R_MAX] - ranges[j][R_MIN];
							} else {
								if (instance.value(j) > ranges[j][R_MAX]) {
									ranges[j][R_MAX] = value;
									ranges[j][R_WIDTH] = ranges[j][R_MAX] - ranges[j][R_MIN];
								}
							}
						}
					}
				}

				return ranges;
			}

		};
	}

	protected void buildKDTree(Instances instances) throws Exception {
		m_DistanceFunction = m_EuclideanDistance = m_override;
		super.buildKDTree(instances);
	}
}
