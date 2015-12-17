package test.moa.classifiers.gpu.zorder;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.DenseInstanceBuffer;
import org.moa.opencl.util.MinMax;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import moa.classifiers.gpu.zorder.ZOrderItem;
import moa.classifiers.gpu.zorder.ZOrderTransform;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class MortonSortTest {
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
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
	
	Instances makeDataset(int num_attr)
	{
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = 0;i < num_attr ; ++i)
			attributes.add( new Attribute(i +""));
		
		return new Instances("a", attributes, 0);
	}
	
	
	@Test
	public void testWithoutNormalize() {
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		int num_attributes = 3;
		int num_rows = 10;
		ZOrderTransform transform = new ZOrderTransform(ctx, num_attributes, num_rows);
		
		Buffer min_values = new Buffer(ctx, 1);
		Buffer max_values = new Buffer(ctx, 1);
		Buffer attribute_map = new Buffer(ctx, 1);
		Instances dataset = makeDataset(num_attributes);
		dataset.setClassIndex(0);
		
		DenseInstanceBuffer instances = new DenseInstanceBuffer(ctx, num_rows, num_attributes);
		instances.begin(Buffer.WRITE);
		for (int i = 0;i <  num_rows; ++i)
		{
			DenseInstance sample = new DenseInstance(num_attributes);
			sample.setValue(0,num_rows - i);
			sample.setValue(1, num_rows - i);
			sample.setValue(1, num_rows - i);
			sample.setDataset(dataset);
			instances.set(sample, i);
		}
		instances.commit();
		ZOrderItem[] order = transform.createZOrder(dataset, instances, min_values, max_values, attribute_map, false);
		for (int i = 0; i < order.length; ++i)
		{
			//order[i].print();
			assertEquals( order[i].offset()/ (num_attributes * (int)DirectMemory.INT_SIZE), num_rows -i -1);
			if (i < order.length -1)
			{
				int result = order[i].compareTo(order[i+1]);
				if (result >= 0)
				{
					order[i].print();
					order[i+1].print();
				}
				assertEquals(result, -1);
			}
				
			
		}
		System.out.println();
		
	}
	
	
	@Test
	public void testWithNormalize() {
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		int num_attributes = 3;
		int num_rows = 10;
		ZOrderTransform transform = new ZOrderTransform(ctx, num_attributes, num_rows);
		Instances dataset = makeDataset(num_attributes);
		dataset.setClassIndex(0);

		
		MinMax minMax = new MinMax(ctx);
		Buffer min_buffer = new Buffer(ctx, dataset.numAttributes() * DirectMemory.DOUBLE_SIZE);
		Buffer max_buffer = new Buffer(ctx, dataset.numAttributes() * DirectMemory.DOUBLE_SIZE);
		
		Buffer attribute_types = new Buffer(ctx,dataset.numAttributes() * DirectMemory.INT_SIZE);
		attribute_types.mapBuffer(Buffer.WRITE);
		attribute_types.writeArray(0, attributeTypes(dataset));
		attribute_types.commitBuffer();
		
		
		DenseInstanceBuffer instances = new DenseInstanceBuffer(ctx, num_rows, num_attributes);
		instances.begin(Buffer.WRITE);
		for (int i = 0;i <  num_rows; ++i)
		{
			DenseInstance sample = new DenseInstance(num_attributes);
			sample.setValue(0,num_rows - i);
			sample.setValue(1, num_rows - i);
			sample.setValue(1, num_rows - i);
			sample.setDataset(dataset);
			instances.set(sample, i);
		}
		instances.commit();
		minMax.fullMinMaxDouble(dataset, instances, min_buffer, max_buffer);
		ZOrderItem[] order = transform.createZOrder(dataset, instances, min_buffer, max_buffer, attribute_types, true);
		for (int i = 0; i < order.length; ++i)
		{
			assertEquals( order[i].offset()/ (num_attributes * (int)DirectMemory.INT_SIZE), num_rows -i -1);
			if (i < order.length -1)
			{
				int result = order[i].compareTo(order[i+1]);
				assertEquals(result, -1);
			}
		}
	}


}
