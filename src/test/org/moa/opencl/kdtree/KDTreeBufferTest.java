package test.org.moa.opencl.kdtree;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.SlidingWindow;

import org.moa.opencl.knn.tree.KDTreeBufferCPU;
import org.viennacl.binding.Context;

import moa.streams.generators.RandomTreeGenerator;
import weka.core.Instance;

public class KDTreeBufferTest {

	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
	@Test
	public void test() {
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		
		int window_size = 10;
		
		RandomTreeGenerator gen = new RandomTreeGenerator();
		gen.prepareForUse();
		
		ArrayList<Instance> pregen = new ArrayList<Instance>();
		for (int i = 0;i < window_size; ++i)
			pregen.add( gen.nextInstance() );
		
		SlidingWindow window = new SlidingWindow(DenseInstanceBuffer.Kind.DOUBLE_BUFFER,ctx,gen.getHeader(), window_size);
		window.begin();
		for (int i = 0 ; i < window_size; ++i)
			window.update(pregen.get(i));
		window.commit();
		
		KDTreeBufferCPU buffer = new KDTreeBufferCPU(gen.getHeader(), ctx,3,window.model());
		buffer.prepareLevel0();
		for (int i = 1 ; i <= 3; ++i)
			buffer.buildCPU(i);
		buffer.dumpTree();
		
	}

}
