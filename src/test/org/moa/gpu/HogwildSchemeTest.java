package test.org.moa.gpu;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Random;

import org.junit.Test;
import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.UnitOfWork;
import org.moa.opencl.sgd.HogwildScheme;
import org.moa.opencl.util.BufHelper;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;

import moa.classifiers.functions.SGDMultiClass;
import moa.core.DoubleVector;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class HogwildSchemeTest {
	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	private Instances prepareDataset(int size) {
		ArrayList<Attribute> lol = new ArrayList<Attribute>();
		ArrayList classValues = new ArrayList();
		classValues.add("one");
		classValues.add("two");
		Attribute classAttribute = new Attribute("test", classValues );
		lol.add(classAttribute);
		for (int i= 1; i < size; ++i)
			lol.add(new Attribute("0"+i));
		Instances lola = new Instances("a", lol,0);
		
		lola.setClassIndex(0);
		return lola;
	}
	
	private SparseInstance makeMasterClone(Instances dataset, double fill) {
		double[] attrs = new double[dataset.numAttributes()];
		SparseInstance mainClone = new SparseInstance(1, attrs);
		for (int j = 0; j < fill * attrs.length; ++j )
			mainClone.setValue(j, 1);

		mainClone.setDataset(dataset);
		return mainClone;
	}
	class ReferenceSGD extends SGDMultiClass {
		public DoubleVector[] getWeights() 
		{
			return m_weights;
		}
	}


	
	public void testScheme() {
		Context ctx = new Context(Context.DEFAULT_MEMORY, null);
		Instances dataset= prepareDataset(790);
		
		Instance mainClone = makeMasterClone(dataset, 1);
		HogwildScheme scheme = null;// new HogwildScheme(ctx, dataset, 1024, 16,0,0,1,0);
	
		
		
		SGDMultiClass sgd = new ReferenceSGD();
		sgd.prepareForUse();
		
		for (int i = 0; i < 600; ++i)
		{
		
			
			System.out.println("commited "+ i);
			sgd.trainOnInstance(mainClone);
			double[] res = sgd.getVotesForInstance(mainClone);

		}
		
		
	}
	
	@Test
	public void testSingleStep()
	{
		Context ctx = new Context(Context.DEFAULT_MEMORY, null);
		Instances dataset= prepareDataset(5);
		Instance mainClone = makeMasterClone(dataset, 1);
		HogwildScheme scheme = null;// new HogwildScheme(ctx, dataset, 1, 1);
		
		
		Instance makeError =(Instance) mainClone.copy();
		for (int i = 0; i < makeError.numAttributes(); ++i)
			makeError.setValue(i, -0.2);
		makeError.setClassValue(0);
		Random rnd = new Random();
		ReferenceSGD sgd = new ReferenceSGD();
		sgd.prepareForUse();
		for (int i = 0; i < 10000; ++i)
		{
			makeError =(Instance) mainClone.copy();
			for (int k = 0; k < makeError.numAttributes(); ++k)
				makeError.setValue(k, rnd.nextDouble());
			makeError.setClassValue(0);
			
			Instance useInstance =i %2 == 0? mainClone : makeError; 
		
		
		
			System.out.println("commited ");
			sgd.trainOnInstance(useInstance);
			
			
			DoubleVector[] weights = sgd.getWeights();

			double[] tau = BufHelper.rb(scheme.getTau());
			double[] small = BufHelper.rb(scheme.getErrorSmall());
			double[] large = BufHelper.rb(scheme.getErrorLarge());
		}
		
		DoubleVector[] weights = sgd.getWeights();
		
		double[] tau = BufHelper.rb(scheme.getTau());
		double[] small = BufHelper.rb(scheme.getErrorSmall());
		double[] large = BufHelper.rb(scheme.getErrorLarge());
		System.out.println();;

	}


}
