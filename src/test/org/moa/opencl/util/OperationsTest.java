package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.DenseInstanceBuffer;
import org.moa.opencl.util.Distance;
import org.moa.opencl.util.MinMax;
import org.moa.opencl.util.Operations;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class OperationsTest {
	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
  
  Context ctx = new Context(Context.DEFAULT_MEMORY, null);
  
  
	
	 private int[] attributeTypes(Instances dataset)
	    {
	        int[] attributeTypes = new int[dataset.numAttributes()];
	         for (int i = 0 ;i < attributeTypes.length; ++i)
	         {
	             if(dataset.attribute(i).isNumeric())
	                 attributeTypes[i] = 0;
	             else
	             if(dataset.attribute(i).isNominal())
	                 attributeTypes[i] = 1;
	             else
	                 attributeTypes[i] = 2;
	         }
	        return attributeTypes;
	    }
	 
	
	@Test
	public void testDoubleToInt() {
		
		Operations operations = new Operations(ctx);
		
		int[] intInit = new int[10];
		double[] init = new double[10];
		for(int i = 0; i < 10 ; ++i)
		{
			init[i] = 0.01 * i;
		}
		Buffer attmap = new Buffer(ctx, 10 * DirectMemory.INT_SIZE);
		Buffer doubleBuffer = new Buffer(ctx, 10 * DirectMemory.DOUBLE_SIZE);
		Buffer intBuffer = new Buffer(ctx, 10 * DirectMemory.INT_SIZE);
		doubleBuffer.mapBuffer(Buffer.WRITE);
		doubleBuffer.writeArray(0, init);
		doubleBuffer.commitBuffer();
		operations.doubleToInt32(doubleBuffer, 
				attmap,
				intBuffer, 1, 10);
		
		intBuffer.mapBuffer(Buffer.READ);
		intBuffer.readArray(0, intInit);
		intBuffer.commitBuffer();
		for (int i = 0 ;i < 10 ; ++i)
		{
			assertEquals(intInit[i], (int)(0.01 * i * 1000000));
		}
	}
	
	@Test 
	public void testNormalize() 
	{
		int rows = 11;
		int nattr = 10;
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = 0; i < nattr ; ++i)
			attributes.add(new Attribute(""+i));
		Instances sampleDataset = new Instances("sample", attributes, 0);
		sampleDataset.setClassIndex(0);
		
		
		Operations operations = new Operations(ctx);
		
		double[] init = new double[nattr];
		
		Buffer min_buffer =  new Buffer(ctx, DirectMemory.DOUBLE_SIZE * 10);
		Buffer max_buffer =  new Buffer(ctx, DirectMemory.DOUBLE_SIZE * 10);
		
		Buffer normalized = new Buffer(ctx, nattr * rows * DirectMemory.DOUBLE_SIZE);
		DenseInstanceBuffer instance_buffer = new DenseInstanceBuffer(ctx, rows, nattr);
		instance_buffer.begin(Buffer.WRITE);
		for (int i = 0 ; i < rows ; ++i)
		{
			for (int j = 0; j < nattr ; ++j)
				init[j] = i;
			DenseInstance inst = new DenseInstance(1.0, init);
			inst.setDataset(sampleDataset);
			instance_buffer.set(inst, i);
		}
		instance_buffer.commit();
		
		MinMax minMax = new MinMax(ctx);
		minMax.fullMinMaxDouble(sampleDataset, instance_buffer, min_buffer, max_buffer);
		
		double[] min_arr = new double[nattr];
		min_buffer.mapBuffer(Buffer.READ);
		min_buffer.readArray(0, min_arr);
		min_buffer.commitBuffer();
		assertArrayEquals(new double[]{0,0,0,0,0,0,0,0,0,0} , min_arr, 0.00001);
		
		
		double[] max_arr = new double[nattr];
		max_buffer.mapBuffer(Buffer.READ);
		max_buffer.readArray(0, max_arr);
		max_buffer.commitBuffer();
		assertArrayEquals(new double[]{0,10,10,10,10,10,10,10,10,10} , max_arr, 0.00001);

		
		Buffer attribute_map = new Buffer(ctx, nattr* DirectMemory.INT_SIZE);
		attribute_map.mapBuffer(Buffer.READ_WRITE);
		attribute_map.writeArray(0, attributeTypes(sampleDataset));
		attribute_map.commitBuffer();
		
		operations.normalize(instance_buffer.attributes(), 
				normalized, 
				min_buffer, max_buffer, attribute_map, 
				 nattr,
				rows);
		
		double[] normalized_arr = new double[nattr * rows];
		normalized.mapBuffer(Buffer.READ);
		normalized.readArray(0, normalized_arr);
		normalized.commitBuffer();
		double[] sample = new double[ rows *nattr];
		for (int i = 0 ; i < rows ; ++i)
		{
			for (int j = 0; j < nattr ; ++j)
				sample[i*(rows-1) + j] = (double)i/ (rows -1);
			sample[i*(rows-1)] = 0;
		}
		assertArrayEquals(sample, normalized_arr, 0.0001);
		
		
		
	}
  
  public static void main(String[] args) 
  {
    OperationsTest t = new OperationsTest() ; 
    for (int i = 0 ; i < 100; ++i)
    {
      t.testDoubleToInt();;
      t.testNormalize();
    }
  }

}
