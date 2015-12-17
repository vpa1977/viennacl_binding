package test.moa.classifiers.gpu.zorder;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.Random;

import org.junit.Test;
import org.moa.gpu.DenseInstanceBuffer;
import org.moa.opencl.util.MinMax;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import moa.classifiers.gpu.zorder.ZOrderItem;
import moa.classifiers.gpu.zorder.ZOrderTransform;
import moa.streams.generators.RandomTreeGenerator;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;

public class ZOrderTransformTest {
	static 
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

	 
	@Test
	public void testCreate() {
		int num_rows = 1000;
		Random rnd = new Random();
		
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		RandomTreeGenerator gen = new RandomTreeGenerator();
		gen.prepareForUse();
		
		DenseInstanceBuffer test_instance = new DenseInstanceBuffer(ctx,  1, gen.getHeader().numAttributes());
		DenseInstanceBuffer buffer = new DenseInstanceBuffer(ctx,  num_rows, gen.getHeader().numAttributes());
		
		MinMax minMax = new MinMax(ctx);
		Buffer min_buffer = new Buffer(ctx, gen.getHeader().numAttributes() * DirectMemory.DOUBLE_SIZE);
		Buffer max_buffer = new Buffer(ctx, gen.getHeader().numAttributes() * DirectMemory.DOUBLE_SIZE);
		
		Buffer m_attribute_types = new Buffer(ctx,gen.getHeader().numAttributes() * DirectMemory.INT_SIZE);
		m_attribute_types.mapBuffer(Buffer.WRITE);
		m_attribute_types.writeArray(0, attributeTypes(gen.getHeader()));
		m_attribute_types.commitBuffer();
		ZOrderTransform transform = new ZOrderTransform(ctx, gen.getHeader().numAttributes(), num_rows);
		Instance sample = null;
		
		for (int i = 0 ;i < 1000; ++i)
		{
			int sampleRow = rnd.nextInt(num_rows); 
			searchTest(num_rows, sampleRow, gen, test_instance, buffer, minMax, min_buffer, max_buffer, m_attribute_types,
					transform, sample);
		}
		
	}


	private void searchTest(int num_rows, int sampleRow, RandomTreeGenerator gen, DenseInstanceBuffer test_instance,
			DenseInstanceBuffer buffer, MinMax minMax, Buffer min_buffer, Buffer max_buffer, Buffer m_attribute_types,
			ZOrderTransform transform, Instance sample) {
		buffer.begin(Buffer.WRITE);
		for (int i = 0; i < num_rows; ++i)
		{
			if (i == sampleRow)
			{
				sample = gen.nextInstance();
				buffer.set(sample,i);
			}
			else
				buffer.set(gen.nextInstance(),i);
		}
		buffer.commit();
		
		
		
		test_instance.begin(Buffer.WRITE);
		test_instance.set(sample, 0);
		test_instance.commit();
		
		
		minMax.fullMinMaxDouble(gen.getHeader(), buffer, min_buffer, max_buffer);
		ZOrderItem[] items = transform.createZOrder(gen.getHeader(), buffer, min_buffer, max_buffer, m_attribute_types, true);
		/*assertEquals(items.length, num_rows);
		ZOrderItem[] clone = new ZOrderItem[items.length];
		System.arraycopy(items, 0, clone, 0, clone.length);
		Arrays.sort(clone);
		for (int i = 0 ; i < clone.length-1 ; ++i)
		{
			if (clone[i].compareTo(clone[i+1]) > 0)
			{
				clone[i].print();
				clone[i+1].print();
			}
			assertTrue(clone[i].compareTo(clone[i+1])  <= 0 );
		}
		for (int i = 0 ; i < items.length-1 ; ++i)
		{
			if (items[i].compareTo(items[i+1]) > 0)
			{
				items[i].print();
				items[i+1].print();
			}
			assertTrue(items[i].compareTo(items[i+1])  <= 0 );
		}

		items[0].print();
		clone[0].print();*/
		//assertArrayEquals(items, clone);
		int position = transform.findPosition(gen.getHeader(), items, 
				test_instance.attributes(), 
				min_buffer, 
				max_buffer, 
				m_attribute_types);
		
		assertTrue(position >= 0);
	}
	
	/*
	@Test
	public void testSingleVsMultiple() 
	{
		int num_rows = 1000;
		Random rnd = new Random();
		
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		RandomTreeGenerator gen = new RandomTreeGenerator();
		gen.prepareForUse();
		
		DenseInstanceBuffer test_instance = new DenseInstanceBuffer(ctx,  1, gen.getHeader().numAttributes());
		DenseInstanceBuffer buffer = new DenseInstanceBuffer(ctx,  num_rows, gen.getHeader().numAttributes());
		
		MinMax minMax = new MinMax(ctx);
		Buffer min_buffer = new Buffer(ctx, gen.getHeader().numAttributes() * DirectMemory.DOUBLE_SIZE);
		Buffer max_buffer = new Buffer(ctx, gen.getHeader().numAttributes() * DirectMemory.DOUBLE_SIZE);
		
		Buffer m_attribute_types = new Buffer(ctx,gen.getHeader().numAttributes() * DirectMemory.INT_SIZE);
		m_attribute_types.mapBuffer(Buffer.WRITE);
		m_attribute_types.writeArray(0, attributeTypes(gen.getHeader()));
		m_attribute_types.commitBuffer();
		ZOrderTransform transform = new ZOrderTransform(ctx, gen.getHeader().numAttributes(), num_rows);
		Instance[] sample = new Instance[num_rows];
		
				
		buffer.begin(Buffer.WRITE);
		for (int i = 0; i < num_rows; ++i)
		{
			sample[i] = gen.nextInstance();
			buffer.set(sample[i],i);
		}
		buffer.commit();
		minMax.fullMinMaxDouble(gen.getHeader(), buffer, min_buffer, max_buffer);
		
		byte[] full_code = transform.produceMortonCode(gen.getHeader(), buffer.attributes(),
				min_buffer, max_buffer, m_attribute_types, num_rows);
		ZOrderItem[] items = new ZOrderItem[ num_rows ];
		for (int i = 0; i < items.length ; ++i)
		{
			items[i] = new ZOrderItem(full_code, (int)(i *gen.getHeader().numAttributes()*DirectMemory.INT_SIZE), i, (int)(gen.getHeader().numAttributes()*DirectMemory.INT_SIZE));
		}
		ZOrderItem[] sorted = new ZOrderItem[ num_rows ];
		System.arraycopy(items, 0, sorted, 0, sorted.length);
		Arrays.sort(sorted);
		
		ZOrderItem[] fullItems = transform.createZOrder(gen.getHeader(), buffer,
				min_buffer, max_buffer, m_attribute_types, true);
		int[] indices = new int[fullItems.length];
		for (int i = 0; i< indices.length; ++i)
		{
			indices[i] = fullItems[i].offset()/ (int)(gen.getHeader().numAttributes()*DirectMemory.INT_SIZE);
		}
		Arrays.sort(indices);
		for (int i = 0; i < indices.length ; ++i)
			assertEquals(i, indices[i]);
		
		for (int i = 0; i < num_rows ; ++i)
		{
			test_instance.begin(Buffer.WRITE);
			test_instance.set(sample[i], 0);
			test_instance.commit();
			byte[] partial_code = transform.produceMortonCode(gen.getHeader(), test_instance.attributes(),
					min_buffer, max_buffer, m_attribute_types, 1);
			byte[] expected = new byte[partial_code.length];
			System.arraycopy(full_code, i * gen.getHeader().numAttributes(), expected, 0, expected.length);
			assertArrayEquals(partial_code, expected);

			
			int index = findCode(fullItems, partial_code);
			if (index < 0)
			{
				System.out.print("fc:");
				for (int k = 0; k < full_code.length; ++k)
					System.out.print(full_code[k] + " ");
				System.out.println();
			}
			assertTrue(index >= 0);
			ZOrderItem sampleItem = new ZOrderItem(partial_code, 0, gen.getHeader().numAttributes(), -1);
			assertEquals(sampleItem.compareTo(items[i]),0);
			int pos = Arrays.binarySearch(sorted, sampleItem);
			assertTrue(pos >= 0);
		}
		
		
		
		
		
		
		
		
		
	
	}


	private int findCode(ZOrderItem[] fullItems, byte[] partial_code) {
	//	System.out.print("_>");
	//	for (int c : partial_code)
	//		System.out.print(c + " ");
	//	System.out.println();
		ZOrderItem item = new ZOrderItem(partial_code, 0,  -1, partial_code.length);
		for (int i = 0; i < fullItems.length ; ++i)
		{
	//		for (int c : fullItems[i].code()) 
	//			System.out.print(c + " ");
	//		System.out.println();
			if (item.equals(fullItems[i]))
				return i;
		}
		return -1;
	}
*/
}
