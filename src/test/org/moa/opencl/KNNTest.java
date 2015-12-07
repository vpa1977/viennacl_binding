package test.org.moa.opencl;

import static org.junit.Assert.*;

import org.junit.Test;

import moa.classifiers.gpu.KNN;
import moa.streams.generators.RandomTreeGenerator;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;

public class KNNTest {
	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
	@Test
	public void testCreate() throws Exception {
		RandomTreeGenerator rtg = new RandomTreeGenerator();
		rtg.prepareForUse();
		
		IBk bk = new IBk(5);
		
		KNN test = new KNN();
		test.neighboursNumber.setValue(5);
		test.prepareForUse();
		bk.buildClassifier(rtg.getHeader());
		bk.setWindowSize(test.slidingWindowSize.getValue());
		for (int i= 0; i <  1024; ++i)
		{
			Instance inst = rtg.nextInstance();
			bk.updateClassifier(inst);
			
			test.trainOnInstance(inst);
		}
		for (int i= 0; i <  2048; ++i)
		{
			Instance inst = rtg.nextInstance();
			double[] result = test.getVotesForInstance(inst);
			double[] wekaResult = bk.distributionForInstance(inst);
			assertArrayEquals(result, wekaResult, 0.00001);
		}
		System.out.println("done");
		

	}

}
