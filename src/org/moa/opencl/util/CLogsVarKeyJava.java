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
  public String program;
  private Buffer m_tmp_log;
	private Parameters params;
	private boolean m_use_value;
	private Parameters m_parameters;
	
	private static class Parameters 
	{
	
		String keyType = "uint";
		String valueType = "uint";
		int reduceWorkGroupSize = 128;
		 int scanWorkGroupSize = 256;//64;
		 int scatterWorkGroupSize = 256; //64
		 int warpSizeSchedule = 64;
		 int warpSizeMem  = 1;
		 int scatterWorkScale = 3;
		 int scanBlocks = 320; 
		 int keySize = 4;
		 int valueSize = 4;
		 int radixBits = 4;
		
		
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
			int radix = 1 << radixBits;
			int scatterSlice = Math.max(warpSizeSchedule,radix);
			int slicesPerWorkGroup = scatterWorkGroupSize / scatterSlice;
			int blocks = (elements + len - 1) / len;
			blocks = roundUp(blocks, slicesPerWorkGroup);
			assert(blocks <= scanBlocks);
			return blocks;
		}

	}
	
	public CLogsVarKeyJava(Context ctx, boolean use_value, String key_type, String value_type)
	{
		m_parameters = new Parameters();
		switch (key_type)
		{
		case "unsigned int":
			m_parameters.keySize = 4;
			break;
		case "unsigned long":
			m_parameters.keySize = 8;
			break;
		default:
			throw new RuntimeException("Unsupported");
		}
		m_parameters.keyType = key_type;
		if (use_value)
		{
			switch (value_type)
			{
			case "int":
			case "unsigned int":
			case "float":
				m_parameters.valueSize = 4;
				break;
				
			case "double":
			case "unsigned long":
			case "long":
				m_parameters.valueSize = 8;
				break;
			default:
				throw new RuntimeException("Unsupported");
			}
			
			m_parameters.valueType = value_type;
		}
		
		m_context = ctx;
		int radix = 1 << m_parameters.radixBits;
		int scatterSlice = Math.max(m_parameters.warpSizeSchedule,radix);
		m_use_value = use_value;
		String program_name = "radixsortcl_hack" + use_value;
		if (!m_context.hasProgram(program_name))
		{
	
			
			StringBuffer defines = new StringBuffer();
			defines.append("#define WARP_SIZE_MEM "+ m_parameters.warpSizeMem).append("\n");
			defines.append("#define WARP_SIZE_SCHEDULE "+ m_parameters.warpSizeSchedule).append("\n");
			defines.append("#define REDUCE_WORK_GROUP_SIZE "+ m_parameters.reduceWorkGroupSize).append("\n");
			defines.append("#define SCAN_WORK_GROUP_SIZE "+ m_parameters.scanWorkGroupSize).append("\n");
			defines.append("#define SCATTER_WORK_GROUP_SIZE "+ m_parameters.scatterWorkGroupSize).append("\n");
			defines.append("#define SCATTER_WORK_SCALE "+ m_parameters.scatterWorkScale).append("\n");
			defines.append("#define SCATTER_SLICE "+ scatterSlice).append("\n");
			defines.append("#define SCAN_BLOCKS "+ m_parameters.scanBlocks).append("\n");
			defines.append("#define RADIX_BITS "+ m_parameters.radixBits).append("\n");
			if (m_parameters.keySize == 8)
				defines.append("#define BASE_KEY_T unsigned long\n");
			else
				defines.append("#define BASE_KEY_T unsigned int\n");
			//defines.append("#define VALUE_TRANSFORM(test)  (as_ulong(test) ^ (-(as_ulong(test) >> 63) | 0x8000000000000000 )) \n");
			
			HashMap<String, String> stringDefines = new HashMap<String,String>();
			stringDefines.put("KEY_T", m_parameters.keyType);
			if (m_use_value)
				stringDefines.put("VALUE_T", m_parameters.valueType);
			
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
			program = defines.append(code).toString();
			m_context.add(program_name, program);
	}
		
		m_histogram = new Buffer(m_context, m_parameters.scanBlocks * radix * DirectMemory.INT_SIZE);

		m_reduce_kernel = m_context.getKernel(program_name, "radixsortReduce_with_raw");
		
		m_uint_reduce_kernel = m_context.getKernel(program_name, "radixsortReduce");
		
		m_scan_kernel = m_context.getKernel(program_name, "radixsortScan");
		m_scan_kernel.set_arg(0, m_histogram);
		
		m_scatter_kernel = m_context.getKernel(program_name,  "radixsortScatter_with_raw");
		m_uint_scatter_kernel =  m_context.getKernel(program_name,  "radixsortScatter");
		m_scatter_kernel.set_arg(1, m_histogram);
			
	}

	public CLogsVarKeyJava(Context ctx, boolean use_value)
	{
		m_parameters = new Parameters();
		m_context = ctx;
		int radix = 1 << m_parameters.radixBits;
		int scatterSlice = Math.max(m_parameters.warpSizeSchedule,radix);
		m_use_value = use_value;
		String program_name = "radixsortcl_" + use_value;
		if (!m_context.hasProgram(program_name))
		{
	
			
			StringBuffer defines = new StringBuffer();
			defines.append("#define WARP_SIZE_MEM "+ m_parameters.warpSizeMem).append("\n");
			defines.append("#define WARP_SIZE_SCHEDULE "+ m_parameters.warpSizeSchedule).append("\n");
			defines.append("#define REDUCE_WORK_GROUP_SIZE "+ m_parameters.reduceWorkGroupSize).append("\n");
			defines.append("#define SCAN_WORK_GROUP_SIZE "+ m_parameters.scanWorkGroupSize).append("\n");
			defines.append("#define SCATTER_WORK_GROUP_SIZE "+ m_parameters.scatterWorkGroupSize).append("\n");
			defines.append("#define SCATTER_WORK_SCALE "+ m_parameters.scatterWorkScale).append("\n");
			defines.append("#define SCATTER_SLICE "+ scatterSlice).append("\n");
			defines.append("#define SCAN_BLOCKS "+ m_parameters.scanBlocks).append("\n");
			defines.append("#define RADIX_BITS "+ m_parameters.radixBits).append("\n");
			HashMap<String, String> stringDefines = new HashMap<String,String>();
			stringDefines.put("KEY_T", m_parameters.keyType);
			if (m_use_value)
				stringDefines.put("VALUE_T", m_parameters.valueType);
			
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
			program = defines.append(code).toString();
			m_context.add(program_name, program);
		}
		
		m_histogram = new Buffer(m_context, m_parameters.scanBlocks * radix * DirectMemory.INT_SIZE);

		m_reduce_kernel = m_context.getKernel(program_name, "radixsortReduce_with_raw");
		
		m_uint_reduce_kernel = m_context.getKernel(program_name, "radixsortReduce");
		
		m_scan_kernel = m_context.getKernel(program_name, "radixsortScan");
		m_scan_kernel.set_arg(0, m_histogram);
		
		m_scatter_kernel = m_context.getKernel(program_name,  "radixsortScatter_with_raw");
		m_uint_scatter_kernel =  m_context.getKernel(program_name,  "radixsortScatter");
		m_scatter_kernel.set_arg(1, m_histogram);
    
  
	}
	
	public void sortFixedBuffer(Buffer keys,  Buffer values, int size)
	{
  
		//
		int maxBits = m_parameters.keySize << 3;
		if (keys.byteSize()/ m_parameters.keySize < size)
			throw new RuntimeException("Key buffer too small");
		if (m_use_value && values.byteSize()/m_parameters.valueSize < size)
			throw new RuntimeException("Value data buffer too small");
		
		
		if (m_tmp_keys  == null || m_tmp_keys.byteSize() < size * m_parameters.keySize)
			m_tmp_keys = new Buffer(m_context, size * m_parameters.keySize);
		if (m_use_value && (m_tmp_values == null || m_tmp_values.byteSize() < size *m_parameters.valueSize))
			m_tmp_values = new Buffer(m_context, size * m_parameters.valueSize);

		Buffer curKeys = keys;
		Buffer curValues = values;
		Buffer nextKeys = m_tmp_keys;
		Buffer nextValues = m_tmp_values;
		
		int blockSize = m_parameters.getBlockSize(size);
		int blocks = m_parameters.getBlocks(size, blockSize);
		for (int firstBit = 0; firstBit < maxBits; firstBit += m_parameters.radixBits)
		{
     
			enqueueReduceUINT(m_histogram,
				curKeys, 
				blockSize, 
				size, 
				firstBit);
			enqueueScan(m_histogram, blocks);
		
			enqueueScatterUINT(nextKeys,
					nextValues, 
					curKeys, 
				curValues, 
				m_histogram, 
				blockSize,
				size, firstBit);
    
     // if (m_context.memoryType() == Context.HSA_MEMORY)
     //   m_context.finishDefaultQueue();

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
		m_context.submitBarrier(false);
		//
		int maxBits = key_size << 3;
		if (keys.byteSize()/DirectMemory.INT_SIZE < size)
			throw new RuntimeException("Key buffer too small");
		if (key_data.byteSize()/key_size < size)
			throw new RuntimeException("Key data buffer too small");
		if (m_use_value && values.byteSize()/DirectMemory.INT_SIZE < size)
			throw new RuntimeException("Value data buffer too small");
		
		
		
		if (m_tmp_keys  == null || m_tmp_keys.byteSize() != size * DirectMemory.INT_SIZE)
		{
			m_tmp_keys = new Buffer(m_context, size * DirectMemory.INT_SIZE);
			
		}
		if (m_use_value && (m_tmp_values == null || m_tmp_values.byteSize() != size * DirectMemory.INT_SIZE))
		{
			m_tmp_values = new Buffer(m_context, size * DirectMemory.INT_SIZE);
		}

		Buffer curKeys = keys;
		Buffer curValues = values;
		Buffer nextKeys = m_tmp_keys;
		Buffer nextValues = m_tmp_values;
		
		int blockSize = m_parameters.getBlockSize(size);
		int blocks = m_parameters.getBlocks(size, blockSize);
		

		for (int firstBit = 0; firstBit < maxBits; firstBit += m_parameters.radixBits)
		{
			enqueueReduce(m_histogram,
				curKeys, 
				key_data, 
				key_size,
				blockSize, 
				size, 
				firstBit);
      //
			enqueueScan(m_histogram, blocks);
		//	int[] hist_cpu = new int[m_parameters.scanBlocks * (1 << m_parameters.radixBits)];
		//	m_histogram.mapBuffer(Buffer.READ);
		//	m_histogram.readArray(0, hist_cpu);
		//	m_histogram.commitBuffer();
			m_context.finishDefaultQueue();
			enqueueScatter(nextKeys,
					nextValues, 
					curKeys, 
				key_data, 
				key_size,
				curValues, 
				m_histogram, 
				blockSize,
				size, firstBit);

			
			
		/*	int[] pre_sort = new int[size];
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
    m_context.submitBarrier(true);
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
		int blocks = m_parameters.getBlocks(elements, len);
		int radix = 1 << m_parameters.radixBits;
		int scatterSlice = Math.max(m_parameters.warpSizeSchedule,radix);
		int slicesPerWorkGroup = m_parameters.scatterWorkGroupSize / scatterSlice;
		int workGroups = blocks / slicesPerWorkGroup;
		m_scatter_kernel.set_global_size(0, m_parameters.scatterWorkGroupSize * workGroups);
		m_scatter_kernel.set_local_size(0,  m_parameters.scatterWorkGroupSize);
		m_scatter_kernel.invoke();
	}

	private void enqueueScan(Buffer histogram, int blocks) {
		m_scan_kernel.set_arg(0, histogram);
		m_scan_kernel.set_arg(1, blocks);
		m_scan_kernel.set_global_size(0, m_parameters.scanWorkGroupSize);
		m_scan_kernel.set_local_size(0, m_parameters.scanWorkGroupSize);
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
		int blocks = m_parameters.getBlocks(size, blockSize);
		m_reduce_kernel.set_global_size(0, blocks* m_parameters.reduceWorkGroupSize);
		m_reduce_kernel.set_local_size(0, m_parameters.reduceWorkGroupSize);
		m_reduce_kernel.invoke();
	}
	
	
	private void enqueueReduceUINT(
			Buffer histogram, 
			Buffer curKeys, 
			int blockSize,
			int size, 
			int firstBit) {
    
/*    int[] wg_items = new int[512];
    m_tmp_log = new Buffer(m_context, 512 * DirectMemory.INT_SIZE);
    m_tmp_log.mapBuffer(Buffer.WRITE);
    m_tmp_log.readArray(0, wg_items);
    m_tmp_log.commitBuffer();
  */  
		m_uint_reduce_kernel.set_arg(0,  histogram);
		m_uint_reduce_kernel.set_arg(1,  curKeys);
		m_uint_reduce_kernel.set_arg(2,  blockSize);
		m_uint_reduce_kernel.set_arg(3,  size);
		m_uint_reduce_kernel.set_arg(4,  firstBit);
    //m_uint_reduce_kernel.set_arg(5, m_tmp_log);
		int blocks = m_parameters.getBlocks(size, blockSize);
		m_uint_reduce_kernel.set_global_size(0, blocks* m_parameters.reduceWorkGroupSize);
		m_uint_reduce_kernel.set_local_size(0, m_parameters.reduceWorkGroupSize);
		m_uint_reduce_kernel.invoke();
    
    
    /*m_tmp_log.mapBuffer(Buffer.READ);
    m_tmp_log.readArray(0, wg_items);
    m_tmp_log.commitBuffer();
            */
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
		int blocks = m_parameters.getBlocks(elements, len);
		int radix = 1 << m_parameters.radixBits;
		int scatterSlice = Math.max(m_parameters.warpSizeSchedule,radix);
		int slicesPerWorkGroup = m_parameters.scatterWorkGroupSize / scatterSlice;
		int workGroups = blocks / slicesPerWorkGroup;
		m_uint_scatter_kernel.set_global_size(0, m_parameters.scatterWorkGroupSize * workGroups);
		m_uint_scatter_kernel.set_local_size(0,  m_parameters.scatterWorkGroupSize);
		m_uint_scatter_kernel.invoke();
            
	}

	
}
