package test.org.moa.gpu;

import static org.junit.Assert.*;

import org.junit.Test;
import org.moa.opencl.sgd.SimpleUpdater;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

public class SimpleUpdaterTest {

	static
	{
		System.loadLibrary("viennacl-java-binding");
	}
	
	@Test
	public void testUpdate() {
		Context ctx = new Context(Context.DEFAULT_MEMORY, null);
		long mem_size  = DirectMemory.DOUBLE_SIZE;
		int num_classes = 3;
		int num_attributes = 5;
		int num_batches = 4;
		SimpleUpdater updater = new SimpleUpdater(ctx, num_attributes,
				num_classes, num_batches,num_batches);

		Buffer gradient_buffer = new Buffer(ctx, num_classes* num_attributes*mem_size); 
		Buffer tau = new Buffer(ctx, num_classes* num_attributes*mem_size);
		tau.mapBuffer(Buffer.WRITE);
		tau.writeArray(0, new double[]{ 1,1,1,1,1,    1,1,1,1,1, 1,1,1,1,1 } );
		tau.commitBuffer();
		
		gradient_buffer.mapBuffer(Buffer.WRITE);
		gradient_buffer.writeArray(0,  new double[]{ 0.7, 1.2, 0.5, 1.4, 1, 
													 1.7, 0.2, 1.5, 0.4, 1, 			
													 1.3, 1.1, 0.5, 1.0, 1.2}
		);
		gradient_buffer.commitBuffer();
		updater.setTau(tau);
		int batch_number = 2;
		updater.applyUpdate(gradient_buffer, batch_number);
		
		
		
		Buffer error_small = updater.getErrorSmall();
		double[] test = new double[num_classes * num_attributes * num_batches];
		error_small.mapBuffer(Buffer.READ);
		error_small.readArray(0,  test);
		error_small.commitBuffer();
		testError(test, batch_number,num_classes, num_attributes, new double[]{
				 0.7, 0.0, 0.5, 0.0, 0, 
				 0.0, 0.2, 0.0, 0.4, 0, 			
				 0.0, 0.0, 0.5, 0.0, 0.0
		});
		Buffer error_large = updater.getErrorLarge();
		
		error_large.mapBuffer(Buffer.READ);
		error_large.readArray(0,  test);
		error_large.commitBuffer();
		testError(test, batch_number,num_classes, num_attributes, new double[]{
				 0.0, 0.2, 0.0, 0.4, 0, 
				 0.7, 0.0, 0.5, 0.0, 0, 			
				 0.3, 0.1, 0.0, 0.0, 0.2
		});
		
		
		gradient_buffer.mapBuffer(Buffer.WRITE);
		gradient_buffer.writeArray(0,  new double[]{  0.3, 0.8, 0.5, 0.6, 0, 
													  0.3, 0.8, 0.5, 0.6, 0, 			
													 0.7, 0.9,  0.5, 0.0, 0.8}
		);
		gradient_buffer.commitBuffer();
		

		updater.applyUpdate(gradient_buffer, batch_number);
		
		error_small = updater.getErrorSmall();
		test = new double[num_classes * num_attributes * num_batches];
		error_small.mapBuffer(Buffer.READ);
		error_small.readArray(0,  test);
		error_small.commitBuffer();
		testError(test, batch_number,num_classes, num_attributes, new double[]{
				 0, 0.0, 0, 0.0, 0, 
				 0, 0, 0.0, 0., 0, 			
				 0.0, 0.0,0, 0.0, 0.0
		});
		
		error_large = updater.getErrorLarge();
		
		error_large.mapBuffer(Buffer.READ);
		error_large.readArray(0,  test);
		error_large.commitBuffer();
		testError(test, batch_number,num_classes, num_attributes, new double[]{
				 0, 0.0, 0, 0.0, 0, 
				 0, 0, 0.0, 0., 0, 			
				 0.0, 0.0,0, 0.0, 0.0
		});
		
		updater.applyWeightsDelta();
		Buffer delta = updater.getWeightsDelta();
		int[] delta_data = new int[num_attributes * num_classes];
		delta.mapBuffer(Buffer.READ);
		delta.readArray(0, delta_data);
		delta.commitBuffer();
		for (int i = 0;i < num_attributes * num_classes; ++i) 
		{
			assertEquals(0,delta_data[i]);
		}
		
		double[] weight_data = updater.getWeights();
		assertArrayEquals(new double[]{1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0},
				weight_data, 0.00001);
				
	}
	
	@Test
	public void updateTau() 
	{
		Context ctx = new Context(Context.DEFAULT_MEMORY, null);
		long mem_size  = DirectMemory.DOUBLE_SIZE;
		int num_classes = 3;
		int num_attributes = 5;
		int num_batches = 2;
		SimpleUpdater updater = new SimpleUpdater(ctx, num_attributes,
				num_classes, num_batches, num_batches);

		Buffer gradient_buffer = new Buffer(ctx, num_classes* num_attributes*mem_size); 
		Buffer tau = new Buffer(ctx, num_classes* num_attributes*mem_size);
		tau.mapBuffer(Buffer.WRITE);
		tau.writeArray(0, new double[]{ 1,1,1,1,1,    1,1,1,1,1, 1,1,1,1,1 } );
		tau.commitBuffer();
		
		gradient_buffer.mapBuffer(Buffer.WRITE);
		gradient_buffer.writeArray(0,  new double[]{ 0.5, 1.5, 0.5, 1.5, 1, 
													 1.5, 0.2, 1.5, 0.4, 1, 			
													 1.3, 1.1, 1.5, 1.0, 1.2}
		);
		gradient_buffer.commitBuffer();
		updater.setTau(tau);
		
		updater.applyUpdate(gradient_buffer, 0);

		gradient_buffer.mapBuffer(Buffer.WRITE);
		gradient_buffer.writeArray(0,  new double[]{ -1.5, 1.5, -0.5, -1.5, -1, 
													 -1.5, -0.2, -1.0, -0.45, -1, 			
													 -1.3, -1.1, -0.5, -1.0, -1.2}
		);
		gradient_buffer.commitBuffer();
		updater.applyUpdate(gradient_buffer, 1);
		updater.updateTau();
		tau = updater.getTau();
		double[] tau_check = new double[num_classes * num_attributes];
		tau.mapBuffer(Buffer.READ);
		tau.readArray(0,tau_check);
		tau.commitBuffer();
		
		assertArrayEquals(new double[]{
				1, 1.5, 0.5, 1.5, 1, 
				1.5, 0.2, 1.25, 0.425, 1,
				1.3, 1.1, 1 , 1,1.2
				
		}, tau_check, 0.00001);
		
		
	}

	private void testError(double[] test, int batch, int num_classes, int num_attributes, double[] ds) {
		int offset = num_classes * num_attributes * batch;
		for (int i = 0; i < ds.length ; ++i)
		{
			assertEquals( test[offset +i], ds[i], 0.0001);
		}
		
	}

}
