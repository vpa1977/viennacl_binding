package test.org.moa.gpu;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.SparseInstanceBuffer;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class SparseInstanceBufferTest {
	
	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}

	@Test
	public void test() {
		int num_attributes = 200;
		int length = 100000;
		
		SparseInstance wekaSparseInstance = new SparseInstance(num_attributes);
		for (int i = 0; i < 50 ; ++i)
			wekaSparseInstance.setValue(i, 0.2);
		Context ctx = new Context(Context.DEFAULT_MEMORY, null);
		Instances lola = prepareDataset();
		SparseInstanceBuffer buffer = new SparseInstanceBuffer(ctx, 10000, num_attributes, 0.3);
		wekaSparseInstance.setDataset(lola);
		buffer.begin(Buffer.WRITE);
		try 
		{
			for (int i = 0; i< length ; ++i)
			{
				Instance iis = (SparseInstance)wekaSparseInstance.copy();
				buffer.append(iis);
			}
			assertFalse(true);
		}
		catch (RuntimeException e ){ assertEquals(e.getMessage(), "Buffer full");}
		buffer.commit();
		
		assertEquals(buffer.rows(), 10000);;
	
		
	}

	private Instances prepareDataset() {
		ArrayList<Attribute> lol = new ArrayList<Attribute>();
		for (int i= 0; i < 100; ++i)
			lol.add(new Attribute("0"+i));
		Instances lola = new Instances("a", lol,0);
		lola.setClassIndex(0);
		return lola;
	}

}
