package test.org.viennacl.binding;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.DenseInstanceBuffer;
import org.moa.opencl.util.BufHelper;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class BufferReleaseTest {

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
	@Test
	public void testCreate() throws InterruptedException
	{
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		Instances set = prepareDataset(1024);
		Instance l = makeMasterClone(set, 1);
		while (true)
		{
			for (int i = 0; i < 100; ++i)
			{
				/*Buffer buf = new Buffer(ctx, 8192* DirectMemory.FLOAT_SIZE);
				float[] test = new float[8192];
				buf.mapBuffer(Buffer.WRITE);
				buf.writeArray(0, test);
				buf.commitBuffer();
				test = BufHelper.rbf(buf);
				*/
				DenseInstanceBuffer buf = new DenseInstanceBuffer(DenseInstanceBuffer.Kind.FLOAT_BUFFER, ctx, 
						1024, 
						1024);
				buf.begin(Buffer.WRITE);
				while (buf.append(l)){}
				buf.commit();
			}
			System.gc();
			Thread.sleep(1);
			
		}
		
	}
}
