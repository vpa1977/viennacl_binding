package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.DenseInstanceBuffer;
import org.moa.opencl.util.MinMax;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import moa.streams.generators.RandomTreeGenerator;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class MinMaxTest {

	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
	@Test
	public void testComputeMinMax() {
		RandomTreeGenerator generator = new RandomTreeGenerator();
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		MinMax minmax = new MinMax(ctx);
		
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = 0;i < 10 ; ++i)
			attributes.add( new Attribute(i +""));
		
		Instances sample_dataset = new Instances("a", attributes, 0);
		sample_dataset.setClassIndex(0);
		
		DenseInstanceBuffer instance_buffer = new DenseInstanceBuffer(ctx, 100, sample_dataset.numAttributes());
		Buffer min_buffer = new Buffer(ctx, sample_dataset.numAttributes() * DirectMemory.DOUBLE_SIZE );
		Buffer max_buffer = new Buffer(ctx, sample_dataset.numAttributes() * DirectMemory.DOUBLE_SIZE );
		instance_buffer.begin(Buffer.WRITE);
		for (int i = 0; i < 100; ++i)
		{
			double[] init = new double[]{i+1,i+2,i+3,i+4,i+5,i+6,i+7,i+8,i+9,i+10};
			DenseInstance sample = new DenseInstance(1.0, init);
			sample.setDataset(sample_dataset);
			
			instance_buffer.set(sample, i);
		}
		instance_buffer.commit();
		
		minmax.fullMinMaxDouble(sample_dataset, instance_buffer, min_buffer, max_buffer);
		min_buffer.mapBuffer(Buffer.READ);
		max_buffer.mapBuffer(Buffer.READ);
		double[] min_vals = new double[10];
		double[] max_vals = new double[10];
		min_buffer.readArray(0, min_vals);
		max_buffer.readArray(0,  max_vals);
		min_buffer.commitBuffer();
		max_buffer.commitBuffer();
		int i = 0;
		assertArrayEquals(new double[]{0,i+2,i+3,i+4,i+5,i+6,i+7,i+8,i+9,i+10}, min_vals, 0.0001);
		i = 99;
		assertArrayEquals(new double[]{0,i+2,i+3,i+4,i+5,i+6,i+7,i+8,i+9,i+10}, max_vals, 0.0001);
		
		
		DenseInstanceBuffer update_buffer = new DenseInstanceBuffer(ctx, 1, sample_dataset.numAttributes());
		
		double[] init = new double[]{0,0,0,0,0,0,0,0,0,1000};
		DenseInstance sample = new DenseInstance(1.0, init);
		sample.setDataset(sample_dataset);
		update_buffer.begin(Buffer.WRITE);
		update_buffer.set(sample, 0);
		update_buffer.commit();
		
		minmax.updateMinMaxDouble(sample_dataset, update_buffer, min_buffer, max_buffer);
		

		min_buffer.mapBuffer(Buffer.READ);
		max_buffer.mapBuffer(Buffer.READ);
		min_buffer.readArray(0, min_vals);
		max_buffer.readArray(0,  max_vals);
		min_buffer.commitBuffer();
		max_buffer.commitBuffer();
		i = 0;
		assertArrayEquals(new double[]{0,0,0,0,0,0,0,0,0,i+10}, min_vals, 0.0001);
		i = 99;
		assertArrayEquals(new double[]{0,i+2,i+3,i+4,i+5,i+6,i+7,i+8,i+9,1000}, max_vals, 0.0001);

		
	}

}
