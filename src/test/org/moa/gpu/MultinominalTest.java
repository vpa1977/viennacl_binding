package test.org.moa.gpu;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.DenseInstanceBuffer;
import org.moa.gpu.SparseInstanceBuffer;
import org.moa.opencl.sgd.Multinominal;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SparseInstance;

public class MultinominalTest {

	static 
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
	private Instances prepareDataset(int size) {
		ArrayList<Attribute> lol = new ArrayList<Attribute>();
		for (int i= 0; i < size; ++i)
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
	public void testLoad() 
	{
		int num_classes = 10;
		int batch_rows = 65535;
		Instances dataset = prepareDataset(790);
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		Multinominal logistic = new Multinominal(ctx, num_classes, batch_rows, 64);
		
		SparseInstanceBuffer.Kind s = SparseInstanceBuffer.Kind.DOUBLE_BUFFER;
		DenseInstanceBuffer dense_buffer = new DenseInstanceBuffer(ctx, batch_rows, dataset.numAttributes());
		
		SparseInstanceBuffer instance_buffer = new SparseInstanceBuffer(s,ctx, batch_rows, dataset.numAttributes(), 1);
		SparseInstance mainClone = makeMasterClone(dataset, 1);
		instance_buffer.begin(Buffer.WRITE);
		dense_buffer.begin(Buffer.WRITE);
		for (int i = 0;i < batch_rows; ++i)
		{
			instance_buffer.append(mainClone);
			dense_buffer.set(mainClone, i);
		}
		instance_buffer.commit();
		dense_buffer.commit();
		
		Buffer weights = new Buffer(ctx, num_classes * dataset.numAttributes() * DirectMemory.DOUBLE_SIZE);
		weights.fill((byte)22);
		Buffer bias = new Buffer(ctx, num_classes  * DirectMemory.DOUBLE_SIZE);
		weights.fill((byte)2);
		
		logistic.computeGradient(dataset, instance_buffer, weights, bias);
		for (int k = 0; k < 1000; ++k)
		{
			/*long start = System.currentTimeMillis();
			for (int i = 0; i< 10; ++i)
				logistic.computeGradient(dataset, instance_buffer, weights, bias);
			logistic.getComputedGradients().mapBuffer(Buffer.READ);
			logistic.getComputedGradients().commitBuffer();
			long end = System.currentTimeMillis();
			
			System.out.println("window time :"+   (end - start)/10.0);
			*/
			long start = System.currentTimeMillis();
			for (int i = 0; i< 10; ++i)
				logistic.computeGradient(dataset, dense_buffer, weights, bias);
			logistic.getComputedGradients().mapBuffer(Buffer.READ);
			logistic.getComputedGradients().commitBuffer();
			long end = System.currentTimeMillis();
			
			System.out.println("dense window time :"+   (end - start)/10.0);

		}
		
	}
	
	@Test
	public void testProduceProduct() {
		int num_classes = 3;
		int batch_rows = 3;
		Instances dataset = prepareDataset(5);
		Context ctx = new Context(Context.Memory.OPENCL_MEMORY, null);
		Multinominal logistic = new Multinominal(ctx, num_classes, dataset.numAttributes(), 1);
		SparseInstanceBuffer instance_buffer = new SparseInstanceBuffer(ctx, batch_rows, dataset.numAttributes(), 1);
		SparseInstance mainClone = makeMasterClone(dataset, 1);
		instance_buffer.begin(Buffer.WRITE);
		for (int i = 0;i < batch_rows; ++i)
			instance_buffer.append(mainClone);
		instance_buffer.commit();
		
		Buffer weights = new Buffer(ctx,num_classes * dataset.numAttributes()* DirectMemory.DOUBLE_SIZE);
		weights.mapBuffer(Buffer.WRITE);
		weights.writeArray(0,  new double[]{1,1,1,1,1, // result 5 
										    .5,.5,.5,.5,.5, // result 2.5
										    0,1,0,0,1 // result 2
		});
		weights.commitBuffer();
		Buffer matrixResult = new Buffer(ctx, num_classes * batch_rows*DirectMemory.DOUBLE_SIZE);
		
		logistic.computeDotProducts
		(instance_buffer.getColumnIndices(),
		 instance_buffer.getRowJumper(),
		 instance_buffer.getElements(), 
		 instance_buffer.getRowBlocks(),
		 instance_buffer.getRowBlockNum(), 
		 instance_buffer.getColumnCount(),
		 batch_rows, 
		 instance_buffer.getRowPostion(), 
		  num_classes, weights,
		  matrixResult);

		long start = System.currentTimeMillis();
		for (int o = 0; o < 100 ; ++o)
		logistic.computeDotProducts
		(instance_buffer.getColumnIndices(),
		 instance_buffer.getRowJumper(),
		 instance_buffer.getElements(), 
		 instance_buffer.getRowBlocks(),
		 instance_buffer.getRowBlockNum(), 
		 instance_buffer.getColumnCount(),
		 batch_rows, 
		 instance_buffer.getRowPostion(), 
		  num_classes, weights,
		  matrixResult);
		
		double[] result = new double[num_classes * batch_rows];
		matrixResult.mapBuffer(Buffer.READ);
		matrixResult.readArray(0,  result);
		matrixResult.commitBuffer();
		long end = System.currentTimeMillis();
		System.out.println("done in " + (end -start));
		assertArrayEquals(new double[]{5, 2.5, 2, 5, 2.5, 2, 5, 2.5, 2} , result, 0.00001);
		
		Buffer bias = new Buffer(ctx, num_classes * DirectMemory.DOUBLE_SIZE );
		double[] class_values = new double[]{0,1,2};
		Buffer clazz_values = new Buffer(ctx, num_classes* DirectMemory.DOUBLE_SIZE);
		clazz_values.mapBuffer(Buffer.WRITE);
		clazz_values.writeArray(0, class_values);
		clazz_values.commitBuffer();
		logistic.computeMultinominalHinge(matrixResult, clazz_values, bias, num_classes, batch_rows);
		matrixResult.mapBuffer(Buffer.READ);
		matrixResult.readArray(0,  result);
		matrixResult.commitBuffer();
		assertArrayEquals(new double[]{0, -1, -1, -1, 0, -1, -1, -1, 0} , result, 0.00001);
		Buffer minibatch_gradients = new Buffer(ctx, num_classes*batch_rows * dataset.numAttributes() * DirectMemory.DOUBLE_SIZE);
		logistic.computeReduceToMinibatch(minibatch_gradients, matrixResult, batch_rows, num_classes, 
				1, dataset.numAttributes(), instance_buffer);
		result = new double[num_classes*batch_rows * dataset.numAttributes()];
		minibatch_gradients.mapBuffer(Buffer.READ);
		minibatch_gradients.readArray(0,  result);
		minibatch_gradients.commitBuffer();
		double[] expected_updates = new double[]
				{
						 0,0,0,0,0,
						-1,-1,-1,-1,-1, 
						-1,-1,-1,-1,-1,
						
						-1,-1,-1,-1,-1,
						 0,0,0,0,0,
						 -1,-1,-1,-1,-1,
						 
						 -1,-1,-1,-1,-1,
						 -1,-1,-1,-1,-1,
						 0,0,0,0,0
				};
		assertArrayEquals(expected_updates, result, 0.00001);
		logistic.computeReduceToMinibatch(minibatch_gradients, matrixResult, batch_rows, num_classes, 
				1, dataset.numAttributes(), instance_buffer);
		minibatch_gradients.mapBuffer(Buffer.READ);
		minibatch_gradients.readArray(0,  result);
		minibatch_gradients.commitBuffer();
		assertArrayEquals(expected_updates, result, 0.00001);
		
	}
	
	public static void main(String[] args)
	{
		new MultinominalTest().testLoad();
	}

}
