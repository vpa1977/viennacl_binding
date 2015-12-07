package test.org.moa.opencl;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.SlidingWindow;
import org.viennacl.binding.Context;

import moa.classifiers.trees.RandomHoeffdingTree;
import moa.streams.generators.RandomTreeGenerator;
import weka.core.DenseInstance;
import weka.core.Instance;


public class SlidingWindowTest {

	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
	@Test
	public void testCreate() {
		int window_size = 10;
		
		RandomTreeGenerator gen = new RandomTreeGenerator();
		gen.prepareForUse();
		
		ArrayList<Instance> pregen = new ArrayList<Instance>();
		for (int i = 0;i < window_size; ++i)
			pregen.add( gen.nextInstance() );
		
		
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		SlidingWindow window = new SlidingWindow(ctx,gen.getHeader(), window_size);
		
		window.begin();
		for (int i = 0 ; i < window_size; ++i)
			window.update(pregen.get(i));
		window.commit();
		
		window.beginRead();
		for (int i = 0 ; i < window_size; ++i)
		{
			Instance test = window.read(i);
			String alpha =pregen.get(i).toString();
			String beta = test.toString();
			assertEquals(alpha, beta);
		}
			
		window.commit();
		
	}

}
