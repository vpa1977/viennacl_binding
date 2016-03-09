package test.org.moa.opencl.util;

import static org.junit.Assert.*;

import java.awt.Dimension;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;

import org.junit.Test;
import org.moa.gpu.DenseInstanceBuffer;
import org.moa.opencl.util.BufHelper;
import org.moa.opencl.util.CLogsSort;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.moa.opencl.util.CLogsVarKeyJava2;
import org.moa.opencl.util.MortonCode;
import org.moa.opencl.util.Operations;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;

import moa.classifiers.gpu.zorder.ProjectedZOrderTransform;
import moa.recommender.dataset.Dataset;
import moa.streams.generators.RandomRBFGenerator;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;
import weka.core.SparseInstance;


public class SingleMortonCodeTest {

	static
	{
		System.loadLibrary("viennacl-java-binding");
	}
  
  public static void main(String[] args) throws Throwable
  {
	  new SingleMortonCodeTest().benchmarkCPUGPU();
  }
  
  public BigInteger interleaveAsString(int[] point) throws Exception {
  	String[] binaryStringValues = new String[point.length];
  	int max_len = -1;
  	for (int i = 0; i < binaryStringValues.length ; ++i)
  	{
  		
  		binaryStringValues[i] = Integer.toBinaryString(point[i]);
  		if (binaryStringValues[i].length() > max_len)
  			max_len = binaryStringValues[i].length();
  	}
  	
  	for (int i = 0; i < binaryStringValues.length ; ++i)
  	{
  		while (binaryStringValues[i].length() < max_len)
  		{
  			binaryStringValues[i] = '0' +binaryStringValues[i]; 
  		}
  	}
  	String res = "";
  	for (int pos = 0; pos <max_len ; ++pos)
  	{
  		for (int i = binaryStringValues.length-1; i >=0 ; --i)
  		{
  			res += binaryStringValues[i].charAt(pos);
  		}
  		/*for (int i = 0; i<binaryStringValues.length ; ++i)
  		{
  			res += binaryStringValues[i].charAt(pos);
  		}*/
  	}
  	
  	BigInteger intValue= new BigInteger(res, 2);
  	
  	return intValue;
  }
  class Unit implements Comparable<Unit>
  {
	  BigInteger lol;
	  int[] point;
	  
	  public String toString() 
	  {
		  String res = "";
		  for (int i : point)
			  res = i + " " + res;
		  return res;
	  }
	@Override
	public int compareTo(Unit o) {
		// TODO Auto-generated method stub
		return lol.compareTo(o.lol);
	}
  }
  
  
  public ArrayList<Unit> createReferencePlot() throws Exception 
  {
	  int num_dimensions = 3;
	  int num_points = 10*10*10;
		int[] source_points = new int[num_dimensions*num_points];
		int[] morton_codes = new int[num_dimensions*num_points];
		int[] cpu_morton_codes = new int[num_dimensions*num_points];
		int offset = 0;
		ArrayList<Unit> seq = new ArrayList<Unit>();
		for (int x = 0; x < 10; ++x)
			for (int y =0 ; y < 10; ++y )
			{
				for (int z =0 ; z < 10; ++z )
				{
					int[] point = new int[]{ x, y, z };
					Unit d = new Unit();
					d.point = point;
					d.lol = interleaveAsString(d.point);
					seq.add(d);
				}
			}
		Collections.sort(seq);;
		
		for (Unit u : seq)
			System.out.println(u);
		
		return seq;
		
	  
  }
  public static String byteToString(byte b) {
	    byte[] masks = { -128, 64, 32, 16, 8, 4, 2, 1 };
	    StringBuilder builder = new StringBuilder();
	    for (byte m : masks) {
	        if ((b & m) == m) {
	            builder.append('1');
	        } else {
	            builder.append('0');
	        }
	    }
	    return builder.toString();
	}
	@Test
	public void testCreate() {
		Context ctx = new Context(Context.DEFAULT_MEMORY, null);
		
		int num_dimensions = 32;
		MortonCode morton = new MortonCode(ctx, num_dimensions);
		Buffer buf = new Buffer(ctx, 2*num_dimensions * DirectMemory.INT_SIZE);
		Buffer output = new Buffer(ctx, 2*num_dimensions * DirectMemory.INT_SIZE);

		
		byte[] cds = BufHelper.rbb(morton.lookupTable());
		int[] point = new int[num_dimensions];
		for (int i = 0; i < point.length; ++i)
			point[i] = 0xFFFFFFFF;
		buf.mapBuffer(Buffer.WRITE);
		buf.writeArray(0,  point);
		buf.commitBuffer();
		
		morton.computeMortonCode(output, buf, 1);
		byte[] code = BufHelper.rbb(output);
		for (int off = 0; off< code.length; ++off)
			System.out.print(byteToString(code[off]));
		System.out.println();
		
	/*	int[] point2 = new int[]{ 65, 2, 75, 65, 2, 75 };
		buf.mapBuffer(Buffer.WRITE);
		buf.writeArray(0,  point2);
		buf.commitBuffer();
		morton.computeMortonCode(output, buf, 2);
		code = BufHelper.rbb(output);
		for (int off = 0; off< code.length; ++off)
			System.out.print(byteToString(code[off]));
		System.out.println();*/
	}
	
	
	private Instances prepareDataset(int size) {
		ArrayList<Attribute> lol = new ArrayList<Attribute>();
		for (int i= 0; i < size; ++i)
			lol.add(new Attribute("0"+i));
		Instances lola = new Instances("a", lol,0);
		lola.setClassIndex(0);
		return lola;
	}
	
	private DenseInstance makeMasterClone(Instances dataset) {
		double[] attrs = new double[dataset.numAttributes()];
		DenseInstance mainClone = new DenseInstance(1, attrs);
		for (int j = 0; j < attrs.length; ++j )
			mainClone.setValue(j, 1);

		mainClone.setDataset(dataset);
		return mainClone;
	}
	
	
	
	@Test
	public void benchmarkCPUGPU()
	{
	//	Context cpu = new Context(Context.Memory.MAIN_MEMORY, null);
		Context opencl = new Context(Context.Memory.OPENCL_MEMORY, null);
 
		System.out.println("batch size\tnum_attributes\topenclTimeMsec\tcpuTimeMsec");
		int num_attributes = 785;
		CLogsVarKeyJava varKeySort = new CLogsVarKeyJava(opencl, false);

		for (int batch_size = 256 ; batch_size < 512000 ; batch_size*= 2)
		{
			ProjectedZOrderTransform transform = new ProjectedZOrderTransform(opencl, varKeySort, num_attributes, batch_size, 64);
			
			Instances dataset = prepareDataset(num_attributes);
			DenseInstanceBuffer openclBuf = new DenseInstanceBuffer(DenseInstanceBuffer.Kind.FLOAT_BUFFER,opencl, batch_size, num_attributes);
			
			
			System.gc();
			Instance inst =makeMasterClone(dataset);
			openclBuf.begin(Buffer.WRITE);
			for (int j = 0; j < batch_size; ++j)
				openclBuf.set(inst, j);
			openclBuf.commit();

			long start = System.nanoTime();
			for (int i = 0; i < 1 ; ++i)
			{
				transform.fillNormalizedData(inst.dataset(), openclBuf);
				transform.createDeviceRandomShiftZOrder(null);
				//openclCode.computeMortonCode(openclOutput, openclBuf.attributes(), batch_size);
				//varKeySort.sort(openclKeys, openclOutput, null, (int)(num_attributes*DirectMemory.INT_SIZE), batch_size);
			}
			opencl.finishDefaultQueue();
			long end = System.nanoTime();
			double openclTimeMsec = (end-start)/1000000;

			start = System.nanoTime();
			/*for (int i = 0; i < 1 ; ++i)
			{

				cpuBuf.begin(Buffer.WRITE);
				for (int j = 0; j < batch_size; ++j)
					cpuBuf.set(inst, j);
				cpuBuf.commit();
				cpuCode.computeMortonCode(cpuOutputOutput, cpuBuf.attributes(), batch_size);
				byte[] result = BufHelper.rbb(cpuOutputOutput);
				ArrayList<Unit> units = new ArrayList<Unit>();
				for (int k = 0; k < batch_size; ++k)
				{
					byte[] next = new byte[(int)(num_attributes*DirectMemory.INT_SIZE)];
					System.arraycopy(result,(int)( k*num_attributes*DirectMemory.INT_SIZE), next,0, (int)(num_attributes*DirectMemory.INT_SIZE));
					BigInteger intBig = new BigInteger(next);
					Unit nextUnit = new Unit();
					nextUnit.lol= intBig;
					nextUnit.point = new int[0];
					units.add(nextUnit);
				}
				Collections.sort(units);
			}*/
			end = System.nanoTime();
			double cpuTimeMsec = (end-start)/1000000;
			System.out.println(batch_size +  "\t" + num_attributes+ "\t"+(openclTimeMsec/1) + "\t"+ (cpuTimeMsec/1));
			
		}
			
		
		
	}
}

