package test.org.moa.gpu;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.UnitOfWork;
import org.moa.opencl.sgd.HogwildScheme;
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
	class c extends SGDMultiClass {
		public DoubleVector[] getWeights() 
		{
			return m_weights;
		}
	}


	@Test
	public void testScheme() {
		Context ctx = new Context(Context.DEFAULT_MEMORY, null);
		Instances dataset= prepareDataset(790);
		
		Instance mainClone = makeMasterClone(dataset, 1);
		HogwildScheme scheme = new HogwildScheme(ctx, dataset, 1024, 16);
		scheme.populate(true);
		
		
		SGDMultiClass sgd = new c();
		sgd.prepareForUse();
		
		for (int i = 0; i < 600; ++i)
		{
			UnitOfWork work = scheme.take();
			work.begin(Buffer.WRITE);
			while (work.append(mainClone)) {}
			work.commit();
			scheme  .put(work);
			System.out.println("commited "+ i);
			sgd.trainOnInstance(mainClone);
			double[] res = sgd.getVotesForInstance(mainClone);

		}
		Buffer weights = scheme.getWeights();
		weights.mapBuffer(Buffer.READ);
		double[] result = new double[790];
		weights.readArray(0,  result);
		weights.commitBuffer();
		for (int i = 0; i< result.length; ++i)
		{
			assertEquals( result[i], 1.0/790.0, 0.00001);
		}
		
	}

}
