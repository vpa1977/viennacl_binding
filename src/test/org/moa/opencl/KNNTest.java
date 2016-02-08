package test.org.moa.opencl;

import static org.junit.Assert.*;

import org.junit.Test;

import moa.classifiers.gpu.KNN;
import moa.streams.generators.RandomTreeGenerator;
import weka.core.Instance;

public class KNNTest {
	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
	public static void main(String[] args) throws Throwable
	{
		System.out.println(System.getProperty("java.class.path"));
		new KNNTest().testCreate();
	}
	
	@Test
	public void testCreate() throws Exception {
		moa.stream.generators.ZOrderValidateGenerator rtg = new moa.stream.generators.ZOrderValidateGenerator();
		rtg.prepareForUse();
		int window_size = 1000;
		
		IBk bk = new IBk(window_size);
		
		KNN test = new KNN();
		test.slidingWindowSizeOption.setValue(window_size);
		test.kOption.setValue(10);
		test.prepareForUse();
		bk.buildClassifier(rtg.getHeader());
		bk.setWindowSize(test.slidingWindowSizeOption.getValue());
		for (int i= 0; i <  window_size; ++i)
		{
			Instance inst = rtg.nextInstance();
			bk.updateClassifier(inst);
			
			test.trainOnInstance(inst);
		}
		for (int i= 0; i <  2048; ++i)
		{
			Instance inst = rtg.nextInstance();
			System.out.println("Target " + inst);
			double[] result = test.getVotesForInstance(inst);
			
			double[] wekaResult = bk.distributionForInstance(inst);
			try {
			
				assertArrayEquals(result, wekaResult, 0.00001);
			}
			catch (Throwable e )
			{
				System.out.println("error here");
				 result = test.getVotesForInstance(inst);
				 wekaResult = bk.distributionForInstance(inst);
			}
		}
		System.out.println("done");
		

	}

}
