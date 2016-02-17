package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.SparseInstanceBuffer;
import org.moa.opencl.sgd.HingeGradient;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class HingeGradientTest {
	static {
		System.loadLibrary("viennacl-java-binding");
	}
	Context context = new Context(Context.DEFAULT_MEMORY, null);

	private Instances prepareDataset(int size) {
		ArrayList<Attribute> lol = new ArrayList<Attribute>();
		for (int i= 0; i < size; ++i)
			lol.add(new Attribute("0"+i));
		Instances lola = new Instances("a", lol,0);
		lola.setClassIndex(0);
		return lola;
	}
	
	
	private SparseInstance makeMasterClone(Instances dataset, int non_zeros) {
		double[] attrs = new double[dataset.numAttributes()];
		SparseInstance mainClone = new SparseInstance(1, attrs);
		for (int j = 0; j < non_zeros; ++j )
			mainClone.setValue(j, 1);

		mainClone.setDataset(dataset);
		return mainClone;
	}
	public static void main(String[] rgs )
	{
		new HingeGradientTest().testCreate();
	}
	
	@Test
	public void testCreate() {
		Instances dataset = prepareDataset(2000);
		Instance instance = makeMasterClone(dataset, 500);
		SparseInstanceBuffer buffer = new SparseInstanceBuffer(context,512, 2000, 0.3);
		buffer.begin(Buffer.WRITE);
		for (int i = 0 ; i < 512; ++i)
			buffer.append(instance);
		buffer.commit();
		
		double[] ww = new double[512];
		for (int i = 0 ; i < ww.length ; ++i)
			ww[i] = 0.000001;
		
		Buffer weights = new Buffer(context, 2000* DirectMemory.DOUBLE_SIZE);
		weights.mapBuffer(Buffer.WRITE);
		weights.writeArray(0, ww);
		weights.commitBuffer();
		
		
		// dotproduct = 0.0005
		// gradient -0.0005
		// loss = 0.9995
		HingeGradient gradient = new HingeGradient(context, 2000, 512);
		Buffer loss = gradient.computeGradient(buffer.classes(), buffer, weights);
		
		double[] lossValues = new double[512];
		loss.mapBuffer(Buffer.READ);
		loss.readArray(0, lossValues);
		loss.commitBuffer();
		for (int i = 0; i < lossValues.length ; ++i)
			assertEquals(lossValues[i], 0.9995, 0.0001);
		
		int[] rows = new int[(int)(buffer.getRowJumper().byteSize()/DirectMemory.DOUBLE_SIZE)];
		buffer.getRowJumper().mapBuffer(Buffer.READ);
		buffer.getRowJumper().readArray(0,  rows);
		buffer.getRowJumper().commitBuffer();
		
		int[] columns = new int[(int) (buffer.getColumnIndices().byteSize()/DirectMemory.DOUBLE_SIZE)];
		buffer.getColumnIndices().mapBuffer(Buffer.READ);
		buffer.getColumnIndices().readArray(0,  columns);
		buffer.getColumnIndices().commitBuffer();
		
		double[] elements = new double [(int)(buffer.getElements().byteSize()/DirectMemory.DOUBLE_SIZE)];
		buffer.getElements().mapBuffer(Buffer.READ);
		buffer.getElements().readArray(0, elements);
		buffer.getElements().commitBuffer();
		
		for (int i = 0; i < columns[rows[rows.length-1]] ; ++i)
			assertEquals("index "+ i, elements[i], -1, 0.0001);
		
		
	}

}
