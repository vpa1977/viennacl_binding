package test.org.viennacl.binding.util;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;

import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;
import org.viennacl.binding.Queue;
/** 
 * Test custom kernel and compare speeds
 * @author john
 *  conclusion - 
 *    -- thread feeder - ok, subject to starvation
 *    -- 
 */
public class DoubleToFloat {
	private Context m_context;
	private Buffer m_src_buffer;
	private Kernel m_kernel;
	private Buffer m_dst_buffer;
	private int m_size;
	private ForkJoinPool service;
	private LinkedBlockingDeque<Buffer> m_source_buffers;
	private LinkedBlockingDeque<Buffer> m_destination_buffers;
	

	public DoubleToFloat(Context ctx, int size)
	{
		if (ctx.memoryType() != Context.MAIN_MEMORY)
		{
			String kernel_src = "__kernel void double_to_float(const uint N, __global double * src, __global  float* dst){ for (uint id = get_global_id(0); id < N; id+= get_global_size(0)) {dst[id] = ((float)src[id]);} }";
			ctx.add("double_to_float", kernel_src);
			m_kernel = ctx.getKernel("double_to_float", "double_to_float");
			Queue data_queue = ctx.createQueue();
			m_source_buffers = new LinkedBlockingDeque<Buffer>();
			m_destination_buffers = new LinkedBlockingDeque<Buffer>();
			int chunk_count= 10;
			int global_size = 16384;
			for (int i = 0; i < (size/global_size)+1 ; ++i )
			{
				m_source_buffers.add(new Buffer(ctx, DirectMemory.DOUBLE_SIZE * global_size, Buffer.WRITE,data_queue));
				m_destination_buffers.add(new Buffer(ctx, DirectMemory.FLOAT_SIZE * global_size, Buffer.READ,data_queue));
			}

			m_kernel.set_arg(0, size);
			//m_kernel.set_arg(1, m_src_buffer);
			//m_kernel.set_arg(2, m_dst_buffer);
			
			service = ForkJoinPool.commonPool();
			int threads = ForkJoinPool.getCommonPoolParallelism();
		}
		m_context = ctx;
		m_size = size;
	}
	
	public void convert(double[] src, float[] dst)
	{
		if (m_size > src.length || m_size > dst.length)
			throw new RuntimeException("converter max needs "+ m_size);
		if (m_context.memoryType() == Context.MAIN_MEMORY)
		{
			for (int i = 0; i < src.length ; ++i)
				dst[i] = ((float)src[i]);
		}
		else
		{
			int global_size = 16384;
			int chunks = src.length /global_size;
			
			while (chunks > 0 && service.awaitQuiescence(10, TimeUnit.MILLISECONDS))
			{
				process_chunk(src, dst, global_size);
				--chunks; 
			}
			while (!service.awaitQuiescence(60, TimeUnit.MINUTES));	
				
		}
	}

	private void process_chunk(double[] src, float[] dst, int global_size) {
		service.submit(()->
		{
			
			try {
				final Buffer src_buffer = m_source_buffers.take();
				src_buffer.mapBuffer(Buffer.WRITE, 0, global_size);
				if (src_buffer.handle() == 0)
					throw new RuntimeException("Map failed");
				DirectMemory.writeArray(src_buffer.handle(), 0, src, global_size);
				src_buffer.commitBuffer();
				service.submit( ()->
				{
					try {
						final Buffer dst_buffer = m_destination_buffers.take();
						m_kernel.set_arg(1, src_buffer);
						m_kernel.set_arg(2, dst_buffer);
						m_kernel.set_global_size(0, 16384);
						m_kernel.set_local_size(0, 256);
						m_kernel.invoke();
						m_source_buffers.add(src_buffer);
						
						service.submit( () ->
						{
							dst_buffer.mapBuffer(Buffer.READ, 0, global_size);
							if (dst_buffer.handle() == 0)
								throw new RuntimeException("Map failed");

							DirectMemory.readArray(dst_buffer.handle(),  dst, global_size);
							dst_buffer.commitBuffer();
							m_destination_buffers.add(dst_buffer);

						});
						
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				 });
				
				
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			
		});
	}
	}
	


