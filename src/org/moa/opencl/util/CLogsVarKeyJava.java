package org.moa.opencl.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;
/** 
 * Radix Sort 
 * UINT key -> 
 * UINT values ->
 * @author john
 *
 */
public class CLogsVarKeyJava  extends AbstractUtil {

	private Context m_context;
	private Buffer m_histogram;
	private Buffer m_tmp_keys;
	private Buffer m_tmp_values;
	private Kernel m_reduce_kernel;
	private Kernel m_scan_kernel;
	private Kernel m_uint_reduce_kernel;
	private Kernel m_uint_scatter_kernel;
	private Kernel m_scatter_kernel;
	private Parameters params;
	private boolean m_use_value;
	
	private static class Parameters 
	{
	
		static String keyType = "uint";
		static String valueType = "uint";
		static int reduceWorkGroupSize = 128;
		static int scanWorkGroupSize = 64;
		static int scatterWorkGroupSize = 64;
		static int warpSizeSchedule = 64;
		static int warpSizeMem  = 1;
		static int scatterWorkScale = 3;
		static int scanBlocks = 320; 
		static int keySize = 4;
		static int valueSize = 4;
		static int radixBits = 4;
		
		
		int roundUp(int x, int y)
		{
		    return (x + y - 1) / y * y;
		}

		
		int getTileSize() 
		{
			return Math.max(reduceWorkGroupSize, scatterWorkScale * scatterWorkGroupSize);
		}

		int getBlockSize(int size) 
		{
			int tileSize = getTileSize();
			return (size + tileSize * scanBlocks - 1) / (tileSize * scanBlocks) * tileSize;
		}
		
		int getBlocks(int elements, int len)
		{
			int radix = 1 << Parameters.radixBits;
			int scatterSlice = Math.max(Parameters.warpSizeSchedule,radix);
			int slicesPerWorkGroup = scatterWorkGroupSize / scatterSlice;
			int blocks = (elements + len - 1) / len;
			blocks = roundUp(blocks, slicesPerWorkGroup);
			assert(blocks <= scanBlocks);
			return blocks;
		}

	}

	public CLogsVarKeyJava(Context ctx, boolean use_value)
	{
		m_context = ctx;
		int radix = 1 << Parameters.radixBits;
		int scatterSlice = Math.max(Parameters.warpSizeSchedule,radix);
		m_use_value = use_value;

		if (!m_context.hasProgram("radixsortcl"))
		{
	
			
			StringBuffer defines = new StringBuffer();
			defines.append("#define WARP_SIZE_MEM "+ Parameters.warpSizeMem).append("\n");
			defines.append("#define WARP_SIZE_SCHEDULE "+ Parameters.warpSizeSchedule).append("\n");
			defines.append("#define REDUCE_WORK_GROUP_SIZE "+ Parameters.reduceWorkGroupSize).append("\n");
			defines.append("#define SCAN_WORK_GROUP_SIZE "+ Parameters.scanWorkGroupSize).append("\n");
			defines.append("#define SCATTER_WORK_GROUP_SIZE "+ Parameters.scatterWorkGroupSize).append("\n");
			defines.append("#define SCATTER_WORK_SCALE "+ Parameters.scatterWorkScale).append("\n");
			defines.append("#define SCATTER_SLICE "+ scatterSlice).append("\n");
			defines.append("#define SCAN_BLOCKS "+ Parameters.scanBlocks).append("\n");
			defines.append("#define RADIX_BITS "+ Parameters.radixBits).append("\n");
			HashMap<String, String> stringDefines = new HashMap<String,String>();
			stringDefines.put("KEY_T", Parameters.keyType);
			if (m_use_value)
				stringDefines.put("VALUE_T", Parameters.valueType);
			
			/* Generate code for upsweep and downsweep. This is done here rather
			* than relying on loop unrolling, constant folding and so on because
			* compilers don't always figure that out correctly (particularly when
			* it comes to an inner loop whose trip count depends on the counter
			* from an outer loop.
			*/
			ArrayList<String> upsweepStmts = new ArrayList<String>();
			ArrayList<String> downsweepStmts = new ArrayList<String>();
			ArrayList<Integer> stops = new ArrayList<Integer>();
			stops.add(1);
			stops.add(radix);
			if (scatterSlice > radix)
				stops.add(scatterSlice);
			stops.add(scatterSlice * radix);
			for (int i = stops.size() - 2; i >= 0; i--)
			{
				int from = stops.get(i + 1);
				int to = stops.get(i);
				if (to >= scatterSlice)
				{
					upsweepStmts.add("upsweepMulti(wg->hist.level1.i, wg->hist.level2.c + "
						+ to + ", " + from + ", " + to + ", lid);");
					downsweepStmts.add("downsweepMulti(wg->hist.level1.i, wg->hist.level2.c + "
						+ to + ", " + from + ", " + to + ", lid);");
				}
				else
				{
					while (from >= to * 4)
					{
						int fromStr =from;
		                int toStr = from / 4;
		                
						boolean forceZero = (from == 4);
						upsweepStmts.add("upsweep4(wg->hist.level2.i + " + toStr + ", wg->hist.level2.c + "
							+ toStr + ", " + toStr + ", lid, SCATTER_SLICE);");
						downsweepStmts.add("downsweep4(wg->hist.level2.i + " + toStr + ", wg->hist.level2.c + "
							+ toStr + ", " + toStr + ", lid, SCATTER_SLICE, "
							+ (forceZero ? "true" : "false") + ");");
						from /= 4;
					}
					if (from == to * 2)
					{
						int fromStr =from;
		                int toStr = from / 2;
						boolean forceZero = (from == 2);
						upsweepStmts.add("upsweep2(wg->hist.level2.s + " + toStr + ", wg->hist.level2.c + "
							+ toStr + ", " + toStr + ", lid, SCATTER_SLICE);");
						downsweepStmts.add("downsweep2(wg->hist.level2.s + " + toStr + ", wg->hist.level2.c + "
							+ toStr + ", " + toStr + ", lid, SCATTER_SLICE, "
							+ (forceZero ? "true" : "false") + ");");
					}
				}
			}
			
			StringBuffer upsweep = new StringBuffer();
			StringBuffer downsweep = new StringBuffer();
			
			upsweep.append("do { ");
			for (int i = 0; i < upsweepStmts.size(); i++)
				upsweep.append(upsweepStmts.get(i));
			upsweep.append(" } while (0)");
			downsweep.append("do { ");
			for (int i = downsweepStmts.size() - 1; i >= 0; i--)
				downsweep.append(downsweepStmts.get(i));
			downsweep.append("} while (0)");
			stringDefines.put("UPSWEEP()", upsweep.toString());
			stringDefines.put("DOWNSWEEP()", downsweep.toString());
	
			
			
			
			Iterator<String> it  = stringDefines.keySet().iterator();
			while (it.hasNext())
			{
				String key = it.next();
				String value = stringDefines.get(key);
				defines.append("#define " + key + " " + value + "\n");
			}
			
			StringBuffer code = loadKernel("radixsort.cl");
			m_context.add("radixsortcl", defines.append(code).toString());
		}
		
		m_histogram = new Buffer(m_context, Parameters.scanBlocks * radix * DirectMemory.INT_SIZE);
		m_reduce_kernel = m_context.getKernel("radixsortcl", "radixsortReduce_with_raw");
		
		m_uint_reduce_kernel = m_context.getKernel("radixsortcl", "radixsortReduce");
		
		m_scan_kernel = m_context.getKernel("radixsortcl", "radixsortScan");
		m_scan_kernel.set_arg(0, m_histogram);
		
		m_scatter_kernel = m_context.getKernel("radixsortcl",  "radixsortScatter_with_raw");
		m_uint_scatter_kernel =  m_context.getKernel("radixsortcl",  "radixsortScatter");
		m_scatter_kernel.set_arg(1, m_histogram);
	}
	
	public void sortUINT(Buffer keys,  Buffer values, int size)
	{
		params = new Parameters();
		
			
		//
		int maxBits = 4 << 3;
		if (keys.byteSize()/DirectMemory.INT_SIZE != size)
			throw new RuntimeException("Key buffer too small");
		if (values.byteSize()/DirectMemory.INT_SIZE != size)
			throw new RuntimeException("Value data buffer too small");
		
		
		if (m_tmp_keys  == null || m_tmp_keys.byteSize() != size * DirectMemory.INT_SIZE)
			m_tmp_keys = new Buffer(m_context, size * DirectMemory.INT_SIZE);
		if (m_tmp_values == null || m_tmp_values.byteSize() != size * DirectMemory.INT_SIZE)
			m_tmp_values = new Buffer(m_context, size * DirectMemory.INT_SIZE);

		Buffer curKeys = keys;
		Buffer curValues = values;
		Buffer nextKeys = m_tmp_keys;
		Buffer nextValues = m_tmp_values;
		
		int blockSize = params.getBlockSize(size);
		int blocks = params.getBlocks(size, blockSize);

		int[] pre_arr = new int[size];
		curKeys.mapBuffer(Buffer.READ);
		curKeys.readArray(0, pre_arr);
		curKeys.commitBuffer();

		for (int firstBit = 0; firstBit < maxBits; firstBit += Parameters.radixBits)
		{
			enqueueReduceUINT(m_histogram,
				curKeys, 
				blockSize, 
				size, 
				firstBit);
			enqueueScan(m_histogram, blocks);
			/*int[] hist_cpu = new int[Parameters.scanBlocks * (1 << Parameters.radixBits)];
			m_histogram.mapBuffer(Buffer.READ);
			m_histogram.readArray(0, hist_cpu);
			m_histogram.commitBuffer();
			*/
			enqueueScatterUINT(nextKeys,
					nextValues, 
					curKeys, 
				curValues, 
				m_histogram, 
				blockSize,
				size, firstBit);
			
/*			int[] pre_sort = new int[size];
			int[] post_sort = new int[size];
			curKeys.mapBuffer(Buffer.READ);
			curKeys.readArray(0, pre_sort);
			curKeys.commitBuffer();

			nextKeys.mapBuffer(Buffer.READ);
			nextKeys.readArray(0, post_sort);
			nextKeys.commitBuffer();
*/
			Buffer tmp = nextKeys;
			nextKeys = curKeys;
			curKeys = tmp;
			
			tmp = nextValues;
			nextValues = curValues;
			curValues = tmp;
			
		}
		if (curKeys != keys)
		{
			curKeys.copyTo(keys);
			curValues.copyTo(values);
		}
	
	}

	
	public void sort(Buffer keys, Buffer key_data, Buffer values, int key_size, int size)
	{
		params = new Parameters();
		
		
		
		//
		int maxBits = key_size << 3;
		if (keys.byteSize()/DirectMemory.INT_SIZE != size)
			throw new RuntimeException("Key buffer too small");
		if (key_data.byteSize()/key_size != size)
			throw new RuntimeException("Key data buffer too small");
		if (m_use_value && values.byteSize()/DirectMemory.INT_SIZE != size)
			throw new RuntimeException("Value data buffer too small");
		
		
		
		if (m_tmp_keys  == null || m_tmp_keys.byteSize() != size * DirectMemory.INT_SIZE)
		{
			m_tmp_keys = new Buffer(m_context, size * DirectMemory.INT_SIZE);
			
		}
		if (m_tmp_values == null || m_tmp_values.byteSize() != size * DirectMemory.INT_SIZE)
		{
			m_tmp_values = new Buffer(m_context, size * DirectMemory.INT_SIZE);
		}

		Buffer curKeys = keys;
		Buffer curValues = values;
		Buffer nextKeys = m_tmp_keys;
		Buffer nextValues = m_tmp_values;
		
		int blockSize = params.getBlockSize(size);
		int blocks = params.getBlocks(size, blockSize);
		

		for (int firstBit = 0; firstBit < maxBits; firstBit += Parameters.radixBits)
		{
			enqueueReduce(m_histogram,
				curKeys, 
				key_data, 
				key_size,
				blockSize, 
				size, 
				firstBit);
			enqueueScan(m_histogram, blocks);
			int[] hist_cpu = new int[Parameters.scanBlocks * (1 << Parameters.radixBits)];
			m_histogram.mapBuffer(Buffer.READ);
			m_histogram.readArray(0, hist_cpu);
			m_histogram.commitBuffer();
			
			enqueueScatter(nextKeys,
					nextValues, 
					curKeys, 
				key_data, 
				key_size,
				curValues, 
				m_histogram, 
				blockSize,
				size, firstBit);
			
			int[] pre_sort = new int[size];
			int[] post_sort = new int[size];
			curKeys.mapBuffer(Buffer.READ);
			curKeys.readArray(0, pre_sort);
			curKeys.commitBuffer();

			nextKeys.mapBuffer(Buffer.READ);
			nextKeys.readArray(0, post_sort);
			nextKeys.commitBuffer();

			Buffer tmp = nextKeys;
			nextKeys = curKeys;
			curKeys = tmp;
			
			tmp = nextValues;
			nextValues = curValues;
			curValues = tmp;
			
		}
		if (curKeys != keys)
		{
			curKeys.copyTo(keys);
			curValues.copyTo(values);
		}
	
	}
	
	private void enqueueScatter(
			Buffer outKeys, 
			Buffer outValues, 
			Buffer inKeys, 
			Buffer inKeysRaw, 
			int raw_key_size,
			Buffer inValues, 
			Buffer histogram, 
			int len, int elements, 
			int firstBit) {
		m_scatter_kernel.set_arg(0,  outKeys);
		m_scatter_kernel.set_arg(1, inKeys);
		m_scatter_kernel.set_arg(2,  inKeysRaw);
		m_scatter_kernel.set_arg(3, raw_key_size);
		m_scatter_kernel.set_arg(4,  histogram);
		m_scatter_kernel.set_arg(5,  len);
		m_scatter_kernel.set_arg(6,  elements);
		m_scatter_kernel.set_arg(7, firstBit);
		if (m_use_value)
		{
			m_scatter_kernel.set_arg(8,  outValues);
			m_scatter_kernel.set_arg(9,  inValues);
		}
		int blocks = params.getBlocks(elements, len);
		int radix = 1 << Parameters.radixBits;
		int scatterSlice = Math.max(Parameters.warpSizeSchedule,radix);
		int slicesPerWorkGroup = Parameters.scatterWorkGroupSize / scatterSlice;
		int workGroups = blocks / slicesPerWorkGroup;
		m_scatter_kernel.set_global_size(0, Parameters.scatterWorkGroupSize * workGroups);
		m_scatter_kernel.set_local_size(0,  Parameters.scatterWorkGroupSize);
		m_scatter_kernel.invoke();
	}

	private void enqueueScan(Buffer histogram, int blocks) {
		m_scan_kernel.set_arg(0, histogram);
		m_scan_kernel.set_arg(1, blocks);
		m_scan_kernel.set_global_size(0, Parameters.scanWorkGroupSize);
		m_scan_kernel.set_local_size(0, Parameters.scanWorkGroupSize);
		m_scan_kernel.invoke();
		
	}

	private void enqueueReduce(
			Buffer histogram, 
			Buffer curKeys, 
			Buffer key_data, 
			int key_size, 
			int blockSize,
			int size, 
			int firstBit) {
		m_reduce_kernel.set_arg(0,  histogram);
		m_reduce_kernel.set_arg(1,  curKeys);
		m_reduce_kernel.set_arg(2,  key_data);
		m_reduce_kernel.set_arg(3,  key_size);
		m_reduce_kernel.set_arg(4,  blockSize);
		m_reduce_kernel.set_arg(5,  size);
		m_reduce_kernel.set_arg(6,  firstBit);
		int blocks = params.getBlocks(size, blockSize);
		m_reduce_kernel.set_global_size(0, blocks* Parameters.reduceWorkGroupSize);
		m_reduce_kernel.set_local_size(0, Parameters.reduceWorkGroupSize);
		m_reduce_kernel.invoke();
	}
	
	
	private void enqueueReduceUINT(
			Buffer histogram, 
			Buffer curKeys, 
			int blockSize,
			int size, 
			int firstBit) {
		m_uint_reduce_kernel.set_arg(0,  histogram);
		m_uint_reduce_kernel.set_arg(1,  curKeys);
		m_uint_reduce_kernel.set_arg(2,  blockSize);
		m_uint_reduce_kernel.set_arg(3,  size);
		m_uint_reduce_kernel.set_arg(4,  firstBit);
		int blocks = params.getBlocks(size, blockSize);
		m_uint_reduce_kernel.set_global_size(0, blocks* Parameters.reduceWorkGroupSize);
		m_uint_reduce_kernel.set_local_size(0, Parameters.reduceWorkGroupSize);
		m_uint_reduce_kernel.invoke();
	}
	
	private void enqueueScatterUINT(
			Buffer outKeys, 
			Buffer outValues, 
			Buffer inKeys, 
			Buffer inValues, 
			Buffer histogram, 
			int len, int elements, 
			int firstBit) {
		m_uint_scatter_kernel.set_arg(0,  outKeys);
		m_uint_scatter_kernel.set_arg(1, inKeys);
		m_uint_scatter_kernel.set_arg(2,  histogram);
		m_uint_scatter_kernel.set_arg(3, (int) len);
		m_uint_scatter_kernel.set_arg(4, (int) elements);
		m_uint_scatter_kernel.set_arg(5, (int)firstBit);
		if (m_use_value)
		{
			m_uint_scatter_kernel.set_arg(6,  outValues);
			m_uint_scatter_kernel.set_arg(7,  inValues);
		}
		int blocks = params.getBlocks(elements, len);
		int radix = 1 << Parameters.radixBits;
		int scatterSlice = Math.max(Parameters.warpSizeSchedule,radix);
		int slicesPerWorkGroup = Parameters.scatterWorkGroupSize / scatterSlice;
		int workGroups = blocks / slicesPerWorkGroup;
		m_uint_scatter_kernel.set_global_size(0, Parameters.scatterWorkGroupSize * workGroups);
		m_uint_scatter_kernel.set_local_size(0,  Parameters.scatterWorkGroupSize);
		m_uint_scatter_kernel.invoke();
	}

	
}
