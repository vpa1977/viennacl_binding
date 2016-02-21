package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;
import org.moa.gpu.SparseInstanceBuffer;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.moa.opencl.util.SparseMatrixOps;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

public class SparseMatrixOpsTest {
	
	static
	{
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
	
	public static void main(String[] args)
	{
		System.out.println(new SparseMatrixOps(new SparseMatrixOpsTest().context).generate("double"));
	}

	
	@Test
	public void testBigMatrix()
	{
		
		//for (int ai = 0;ai < 5; ++ai)
		{
		
		SparseInstanceBuffer buffer = new SparseInstanceBuffer(context, 15000, 5000, 0.2);
		SparseMatrixOps mm = new SparseMatrixOps(context);
	
		Instances dataset = prepareDataset(5000);
		SparseInstance mainClone = makeMasterClone(dataset);
		buffer.begin(Buffer.WRITE);
		for (int i = 0 ;i < 15000; ++i)
		{
			SparseInstance in = (SparseInstance) mainClone.copy();
			buffer.append(in);
		}
		//mm.m_mult.set_local_size(0, 256);
		buffer.commit();
	//	mm.m_mult.set_local_size(0, 256);
		int count = buffer.getRowBlockNum();;
		
		int[] test_row_blocks = new int[count+1];
		buffer.getRowBlocks().mapBuffer(Buffer.READ);
		buffer.getRowBlocks().readArray(0, test_row_blocks);
		buffer.getRowBlocks().commitBuffer();
		
		int[] columns = new int[(int)(buffer.getElements().byteSize()/DirectMemory.DOUBLE_SIZE)];
		buffer.getColumnIndices().mapBuffer(Buffer.READ);
		buffer.getColumnIndices().readArray(0, columns);
		buffer.getColumnIndices().commitBuffer();
		
		int[] rows = new int[(int)buffer.rows()+1];
		buffer.getRowJumper().mapBuffer(Buffer.READ);
		buffer.getRowJumper().readArray(0, rows);
		buffer.getRowJumper().commitBuffer();
		
		
		Buffer weights = new Buffer(context, 15000 * DirectMemory.DOUBLE_SIZE);
		weights.mapBuffer(Buffer.WRITE);
		for (int i = 0 ; i < 15000; ++i)
			weights.write(i*DirectMemory.DOUBLE_SIZE, 1);
		weights.commitBuffer();
		
		Buffer result = new Buffer(context, 15000 * DirectMemory.DOUBLE_SIZE);
		//for (int kk = 0; kk < 100; ++kk)
			mm.mult(buffer, weights, result);
		
		result.mapBuffer(Buffer.READ);
		for (int i = 0 ; i < 15000; ++i)
		{
			double val = result.read(i * DirectMemory.DOUBLE_SIZE);
			
			assertEquals(val, 3000, 0.001);
		}
		result.commitBuffer();
		}
	}

	private SparseInstance makeMasterClone(Instances dataset) {
		double[] attrs = new double[5000];
		SparseInstance mainClone = new SparseInstance(1, attrs);
		for (int j = 0; j < 3000; ++j )
			mainClone.setValue(j, 1);

		mainClone.setDataset(dataset);
		return mainClone;
	}
	
	@Test
	public void testColumnSum() 
	{
		SparseInstanceBuffer buffer = new SparseInstanceBuffer(context, 5, 5, 1);
		SparseMatrixOps mm = new SparseMatrixOps(context);
		Instances dataset = prepareDataset(5);
		SparseInstance newObject = new SparseInstance(1,new double[]{0,0,0,0,0});
		newObject.setDataset(dataset);
		buffer.begin(Buffer.WRITE);
		for (int i = 0 ;i < 5; ++i)
		{
			SparseInstance in = (SparseInstance) newObject.copy();
			in.setValue(i, 1);
			buffer.append(in);
		}
		buffer.commit();
		
		Buffer result  = new Buffer(context, 5 * DirectMemory.DOUBLE_SIZE);
		mm.columnSum(buffer, result);
		context.finishDefaultQueue();
		
		result.mapBuffer(Buffer.READ);
		double[] r = new double[5];
		result.readArray(0,r);;
		result.commitBuffer();
		assertArrayEquals(r,  new double[]{1,1,1,1,1}, 0.0001);
	}
	
	
	@Test 
	public void testBigColumnReduce()
	{
		SparseInstanceBuffer buffer = new SparseInstanceBuffer(context, 5, 3000, 1);
		SparseMatrixOps mm = new SparseMatrixOps(context);
		Instances dataset = prepareDataset(3000);
		SparseInstance newObject = new SparseInstance(1,new double[3000]);
		newObject.setDataset(dataset);
		buffer.begin(Buffer.WRITE);
		for (int i = 0 ;i < 5; ++i)
		{
			SparseInstance in = (SparseInstance) newObject.copy();
			int set = 2500 - i * 500;
			for (int j = set ; j< 3000; ++j)
				in.setValue(j, 1);	
			buffer.append(in);
		}
		buffer.commit();
		
		Buffer result  = new Buffer(context, 3000 * DirectMemory.DOUBLE_SIZE);
		mm.columnSum(buffer, result);
		
		double[] r = new double[3000];
		result.mapBuffer(Buffer.READ);
		result.readArray(0,r);;
		result.commitBuffer();
		for (int i = 0; i < 3000; ++i)
		{
			int expect = 1 + i / 500;
			assertEquals("item " + i, (double)expect, r[i], 0.0001);
			
		}
		
		
	}
	
	
	@Test
	public void testReductionSpeed()
	{
		Instances dataset = prepareDataset(32768);
		int row_len = 32768;
		
		Buffer result  = new Buffer(context, row_len * DirectMemory.DOUBLE_SIZE);
		double[] full = new double[row_len];
		
		for (int i = 0; i < full.length/8 ; ++i)
		{
			full[i] = 1;
		}
		
		SparseMatrixOps mm = new SparseMatrixOps(context);
		SparseInstance newObject = new SparseInstance(1,full);
		newObject.setDataset(dataset);
		for (int total_rows = 1; total_rows < 2048; ++total_rows)
		{
			SparseInstanceBuffer buffer = new SparseInstanceBuffer(context, total_rows, row_len, 1);
		
			buffer.begin(Buffer.WRITE);
			for (int row = 0; row < total_rows; ++row)
				buffer.append((Instance)newObject.copy());
			buffer.commit();
			
			
			long start = System.nanoTime();
			for (int reps = 0; reps < 1000; ++reps)
				mm.columnSum(buffer, result);
			context.finishDefaultQueue();
			long end = System.nanoTime();
			long msecs = (end - start)/1000000;
			
			double[] colu =  new double[total_rows];
			start  = System.nanoTime();
			for (int reps = 0; reps < 1000; ++reps)
			for (int i = 0; i < total_rows; ++i)
				for (int j = full.length-1; j>=0; --j)
				colu[i] += full[j];
			
			end = System.nanoTime();
			long msecs_cpu = (end - start)/1000000;
			
			System.out.println(total_rows + "\t" + msecs + "\t" +msecs_cpu);
		}
		
		
	}
	
	@Test
	public void testMult() {
		
		
		SparseInstanceBuffer buffer = new SparseInstanceBuffer(context, 5, 5, 1);
		SparseMatrixOps mm = new SparseMatrixOps(context);
		Instances dataset = prepareDataset(5);
		SparseInstance newObject = new SparseInstance(1,new double[]{0,0,0,0,0});
		newObject.setDataset(dataset);
		buffer.begin(Buffer.WRITE);
		for (int i = 0 ;i < 5; ++i)
		{
			SparseInstance in = (SparseInstance) newObject.copy();
			in.setValue(i, 1);
			buffer.append(in);
		}
		buffer.commit();
		
		Buffer weights = new Buffer(context, 5 * DirectMemory.DOUBLE_SIZE);
		Buffer result = new Buffer(context, 5 * DirectMemory.DOUBLE_SIZE);
		weights.mapBuffer(Buffer.WRITE);
		weights.writeArray(0,  new double[]{1,1,1,1,1});
		weights.commitBuffer();
		
		assertEquals(buffer.getRowBlockNum(), 1);
		Buffer b = buffer.getRowBlocks();
		int[] blocks = new int[5];
		b.mapBuffer(Buffer.READ);
		b.readArray(0, blocks);
		b.commitBuffer();
		assertEquals(blocks[0], 0);
		assertEquals(blocks[1], 5);
		assertEquals(blocks[2], 0);
		
		int[] rows = new int[buffer.rows()+1];
		Buffer row_jumper = buffer.getRowJumper();
		row_jumper.mapBuffer(Buffer.READ);
		row_jumper.readArray(0,  rows);
		row_jumper.commitBuffer();
		
		int[] columns = new int[(int)(buffer.getElements().byteSize()/DirectMemory.DOUBLE_SIZE)];
		Buffer col = buffer.getColumnIndices();
		col.mapBuffer(Buffer.READ);
		col.readArray(0,  columns);
		col.commitBuffer();
		
		mm.mult(buffer, weights, result);
		
		double[] test = new double[5];
		result.mapBuffer(Buffer.READ);
		result.readArray(0, test);
		result.commitBuffer();
		assertArrayEquals(test, new double[]{1,1,1,1,1}, 0.001);
	}
	
	

}
