package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Random;

import org.junit.Test;
import org.moa.gpu.DenseInstanceBuffer;
import org.moa.opencl.util.Distance;
import org.moa.opencl.util.MinMax;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import moa.classifiers.lazy.neighboursearch.EuclideanDistance;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class DistanceTest {

	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
	@Test
	public void testCreate() {
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		Distance d = new Distance(ctx);
		assertTrue(d != null);
		int attributes = 3;
		ArrayList<Attribute> att = new ArrayList<Attribute>();
		for (int i = 0; i< attributes; ++i)
			att.add( new Attribute(""+i));
		Instances instances = new Instances("aaa", att, 0);
		instances.setClassIndex(0);
		Random rnd = new Random();
		int rows = 10;
		
		Instance test_instance = new DenseInstance(3);
		test_instance.setDataset(instances);
		for (int k = 0; k < attributes; ++k)
			test_instance.setValue(k, rnd.nextDouble());

		
		
		DenseInstanceBuffer dib = new DenseInstanceBuffer(ctx, rows+1, attributes);
		dib.begin(Buffer.WRITE);
		for (int i = 0; i < rows; ++i)
		{
			Instance next = new DenseInstance(3);
			next.setDataset(instances);
		
			for (int k = 0; k < attributes; ++k)
				next.setValue(k, rnd.nextDouble());
			instances.add(next);
			dib.set(next,i);
		}
		dib.set(test_instance, rows);
		dib.commit();
		
		DenseInstanceBuffer tib = new DenseInstanceBuffer(ctx, 1, attributes);
		tib.begin(Buffer.WRITE);
		instances.add(test_instance);
		tib.set(test_instance, 0);
		tib.commit();
		
		MinMax mm = new MinMax(ctx);
		
		Buffer min_buffer = new Buffer(ctx, DirectMemory.DOUBLE_SIZE * attributes * 2);
		Buffer max_buffer = new Buffer(ctx, DirectMemory.DOUBLE_SIZE * attributes * 2);
		Buffer result = new Buffer(ctx, DirectMemory.DOUBLE_SIZE * (rows+1));
		Buffer types = new Buffer(ctx, DirectMemory.DOUBLE_SIZE * attributes);
		int[] itypes = new int[attributes];
		types.mapBuffer(Buffer.WRITE);
		types.writeArray(0, itypes);
		types.commitBuffer();
		
		mm.fullMinMaxDouble(instances, dib, min_buffer, max_buffer);
		d.squareDistance(instances, tib, dib, min_buffer, max_buffer, types, result);
		
		EuclideanDistance reference = new EuclideanDistance(instances);
		
		
		double[] results = new double[rows+1];
		result.mapBuffer(Buffer.READ);
		result.readArray(0, results);
		result.commitBuffer();
		double[] ref_results = new double[rows+1];
		for (int i = 0; i  < instances.numInstances(); ++ i)
		{
			double ref_dist = reference.distance(instances.get(i),test_instance);
			ref_results[i] = ref_dist;
			results[i]  = Math.sqrt(results[i]);
		}
		assertArrayEquals(ref_results, results, 0.0001);
		
	}

}

