package org.viennacl.binding;

import sun.misc.Unsafe;

public class Buffer {
	public static final int WRITE = 1;
	public static final int READ = 2;
	public static final int READ_WRITE = 3;
	private Context m_context;
	private long m_context_ptr;
	private long m_byte_size;
	private int m_memory_type;
	private int m_mode;
	private Queue m_queue;
	private boolean m_mapped;
	private Exception alloc_location;
	
	public Buffer(Context ctx, long size, int mode, Queue q)
	{
		m_mode = mode;
		m_context = ctx;
		m_memory_type = ctx.memoryType();
		m_byte_size = size;
		m_queue = q;
		m_mapped = false;
		m_context_ptr = ctx.getNativeContext();
		alloc_location = new Exception();
		allocate();
	}

	
	public Buffer(Context ctx, long size, int mode)
	{
		this(ctx, size, mode, null);
		
	}
	public Buffer(Context ctx, long size)
	{
		this(ctx, size, READ_WRITE, null);
	}
	
	public long handle()
	{
		return m_cpu_memory;
	}
	
	@Override
	protected void finalize() throws Throwable {
		if (m_cpu_memory != 0)
		{
			System.out.println("Releasing a mapped buffer. Should not");
			alloc_location.printStackTrace();
		}
		if (m_context == null)
			System.out.println("Releasing a buffer without context. Should not");
		release(m_context_ptr);
	}
	
	
	public synchronized void  mapBuffer(int mode)
	{
		if (m_mapped)
			throw new RuntimeException("Buffer is being accessed");
		if (m_context.memoryType() == Context.HSA_MEMORY)
			m_context.finishDefaultQueue();
		map(mode);
		m_mapped = true;
	}
	
	public synchronized void  mapBuffer(int mode, long offset, long length) 
	{
		if (m_mapped)
			throw new RuntimeException("Buffer is being accessed");
		if (offset + length > byteSize())
			throw new RuntimeException("Buffer overrun");
		map(mode, offset, length);
		m_mapped = true;
	}
	
	public  synchronized void  commitBuffer()
	{
		commit();
		m_mapped = false;
	}
	
	public long byteSize() {
		return m_byte_size;
	}
	
	
	
	public void write(long handle, double value)
	{
		runtimeCheck(handle);
		
		DirectMemory.write(m_cpu_memory + handle, value);
		
	}
	
	public void set(long handle, long size, byte value)
	{
		runtimeCheck(handle);
		DirectMemory.set(m_cpu_memory+handle, size, value);
	}

	
	public double read(long handle)
	{
		runtimeCheck(handle);

		return DirectMemory.read(m_cpu_memory + handle);
	}
	
	public  void writeInt(long handle, int value) {
		runtimeCheck(handle);
		DirectMemory.writeInt(m_cpu_memory+handle, value);
	}
	
	public int readInt(long handle)
	{
		runtimeCheck(handle);
		return DirectMemory.readInt(m_cpu_memory+handle);
	}
	
	public byte readByte(int handle) {
		runtimeCheck(handle);
		return DirectMemory.readByte(m_cpu_memory+handle);
	}



	private void runtimeCheck(long handle) {
		if (handle > byteSize())
			throw new RuntimeException("Buffer overrun");
		if (m_cpu_memory==0)
			throw new RuntimeException("Null pointer");
	}
	

	public  void writeLong(long handle, long data) {
		runtimeCheck(handle);
		DirectMemory.writeLong(m_cpu_memory + handle, data);
	}
	
	public void writeByte(long handle, byte res) {
		runtimeCheck(handle);
		DirectMemory.writeByte(m_cpu_memory + handle, res);
	}


	public void writeArray(long writeIndex, double[] data) {
		long len = (writeIndex + data.length) * DirectMemory.DOUBLE_SIZE;
		runtimeCheck(len);
		DirectMemory.writeArray(m_cpu_memory, writeIndex, data);
	}
	
	public void writeArray(long m_buffer, long writeIndex, double[] data, long length) {
		long len =( writeIndex + length) * DirectMemory.DOUBLE_SIZE;
		runtimeCheck(len);
		if (length > data.length)
			throw new RuntimeException("length >>> data.length");
		DirectMemory.writeArray(m_cpu_memory, writeIndex, data, length);
	}
	
	
	public void writeArray(long writeIndex, int[] data)
	{
		long len = (writeIndex + data.length) * DirectMemory.INT_SIZE;
		runtimeCheck(len);
		DirectMemory.writeArray(m_cpu_memory, writeIndex, data);
	}
	
	public void writeArray(int writeIndex, long[] data) {
		long len = (writeIndex + data.length) * DirectMemory.LONG_SIZE;
		runtimeCheck(len);
		DirectMemory.writeArray(m_cpu_memory, writeIndex, data);
	}
	
	public void writeArray(int writeIndex, float[] data) {
		long len = (writeIndex + data.length) * DirectMemory.FLOAT_SIZE;
		runtimeCheck(len);
		DirectMemory.writeArray(m_cpu_memory, writeIndex, data);
	}
	
	public void writeArray(int writeIndex, byte[] data) {
		long len = writeIndex + data.length;
		runtimeCheck(len);
		DirectMemory.writeArray(m_cpu_memory, writeIndex, data);
		
	}


	
	public long readLong(long handle) {
		runtimeCheck(handle);
		return DirectMemory.readLong(m_cpu_memory + handle);
	}

	public void readArray(long handle, float[] dst) {
		runtimeCheck(handle + dst.length * DirectMemory.FLOAT_SIZE);
		DirectMemory.readArray(m_cpu_memory + handle, dst);
	}
	
	public void readArray(long handle, int[] dst) {
		runtimeCheck(handle + dst.length * DirectMemory.INT_SIZE);
		DirectMemory.readArray(m_cpu_memory + handle, dst);
	}
	
	public void readArray(long handle,  long[] dst) {
		runtimeCheck(handle + dst.length * DirectMemory.LONG_SIZE);
		DirectMemory.readArray(m_cpu_memory + handle, dst);
	}
	
	public void readArray(int handle, byte[] dst) {
		runtimeCheck(handle + dst.length);
		DirectMemory.readArray(m_cpu_memory + handle, dst);
	}

	
	public void readArray(long handle, double[] dst) {
		runtimeCheck(handle + dst.length * DirectMemory.DOUBLE_SIZE);
		DirectMemory.readArray(m_cpu_memory + handle, dst);
	}

	
	public  void readArray(long handle, int i, float[] dst, long length) {
		runtimeCheck(handle + length * DirectMemory.FLOAT_SIZE);
		if (length > dst.length)
			throw new RuntimeException("length >>> data.length");
		DirectMemory.readArray(m_cpu_memory + handle, dst, length);
	}
	
	
	
	public Buffer copy()
	{
		Buffer my_clone = new Buffer(m_context, m_byte_size, m_mode, m_queue);
		native_copy(my_clone);
		return my_clone;
	}
	
	public void copyTo(Buffer target)
	{
		if (target.byteSize() < byteSize())
			throw new RuntimeException("Target is too small");
		native_copy(target);
	}


	public void checkedFill(byte b, long length)
	{
		if (length > byteSize())
			throw new RuntimeException("Buffer too small");
		fill(b,length);
	}

	public void copyTo(Buffer target, int offset) {
		if (target.byteSize() - offset  < byteSize())
			throw new RuntimeException("Buffer too small");
		native_copy_to(target, offset);
	}
	
	/* map all data to cpu */
	public native void fill(byte b);
	public native void fill(byte b, long length);
	private native void map(int mode);
	private native void map(int mode, long offset, long length);
	private native void commit();
	
	/* gpu->gpu buffer copy */
	private native void native_copy(Buffer cloned);
	private native void native_copy_to(Buffer target, long target_offset);
	private native void native_copy(Buffer cloned, long src_offset, long length);
	private native void allocate();
	// use with care - will work only for opencl implementations.
	public synchronized native void release(); // public release of the buffer data
	private synchronized native void release(long context_ptr); // private release from finalize, using cached context
	
	// native vector context - contains handles to GPU memory
	private long m_native_context;
	// readable/writable CPU memory;
	private long m_cpu_memory;



}
