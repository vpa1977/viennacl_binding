package org.viennacl.binding;

import java.lang.reflect.Constructor;

import sun.misc.Unsafe;

public class DirectMemory {
	
	public static long DOUBLE_SIZE = 8; // todo fix in native code.
	public static long INT_SIZE = 4; // todo fix in native code.
	public static final long LONG_SIZE = 8; // todo fix in native code
	public static final int FLOAT_SIZE = 4;
	private static Unsafe m_direct_memory = getUnsafe();
	
	public static long allocate(long length)
	{
		long handle =m_direct_memory.allocateMemory(length);
		return handle;
	}
	
	
	public static void free(long handle)
	{
		m_direct_memory.freeMemory(handle);
		
	}
	
	public static void write(long handle, double value)
	{
		if (handle == 0)
			throw new RuntimeException("Attempt to access null pointer");
		//System.out.println("Writing to "+ handle + " 1 double" );
		m_direct_memory.putDouble(handle, value);
		//System.out.println("Data write complete" );
	}
	
	public static void set(long handle, long size, byte value)
	{
		if (handle == 0)
			throw new RuntimeException("Attempt to access null pointer");
		m_direct_memory.setMemory(handle, size, value);
	}

	
	public static double read(long handle)
	{
		if (handle == 0)
			throw new RuntimeException("Attempt to access null pointer");
		//System.out.println("Read from "+ handle + " 1 double" );
		return m_direct_memory.getDouble(handle);
	}
	
	public static void writeInt(long handle, int value) {
		if (handle == 0)
			throw new RuntimeException("Attempt to access null pointer");
		m_direct_memory.putInt(handle, value);
	}
	
	public static int readInt(long handle)
	{
		if (handle == 0)
			throw new RuntimeException("Attempt to access null pointer");
		return m_direct_memory.getInt(handle);
	}
	

	public static void writeLong(long handle, long data) {
		if (handle == 0)
			throw new RuntimeException("Attempt to access null pointer");
		m_direct_memory.putLong(handle,  data);
		
	}

	public static void writeArray(long m_buffer, long writeIndex, double[] data) {
		if (m_buffer == 0)
			throw new RuntimeException("Attempt to access null pointer");

		//System.out.println("Writing to "+ m_buffer + " offset " + writeIndex + " data len " + data.length);
		long read_from =m_direct_memory.ARRAY_DOUBLE_BASE_OFFSET;
		long write_to =   writeIndex * DOUBLE_SIZE;
		m_direct_memory.copyMemory(data, read_from, null, m_buffer+write_to, data.length*m_direct_memory.ARRAY_DOUBLE_INDEX_SCALE );
	}
	
	public static void writeArray(long m_buffer, long writeIndex, double[] data, long length) {
		if (m_buffer == 0)
			throw new RuntimeException("Attempt to access null pointer");
		long read_from =m_direct_memory.ARRAY_DOUBLE_BASE_OFFSET;
		long write_to =   writeIndex * DOUBLE_SIZE;
		m_direct_memory.copyMemory(data, read_from, null, m_buffer+write_to, length*m_direct_memory.ARRAY_DOUBLE_INDEX_SCALE );
	}
	
	/*public static void readArray(long m_buffer, double[] data)
	{
		long write_to =m_direct_memory.ARRAY_DOUBLE_BASE_OFFSET;
		Object helperArray[] 	= new Object[1];
		helperArray[0] 		= data;
		long baseOffset 		= m_direct_memory.arrayBaseOffset(Object[].class);
		long addressOfObject	= m_direct_memory.getLong(helperArray, baseOffset);
		m_direct_memory.copyMemory(m_buffer,addressOfObject + write_to, data.length * DOUBLE_SIZE); 
	}*/
	
	public static void writeArray(long m_buffer, long writeIndex, int[] data)
	{
		if (m_buffer == 0)
			throw new RuntimeException("Attempt to access null pointer");
		long read_from = m_direct_memory.ARRAY_INT_BASE_OFFSET;
		long write_to =   writeIndex * DOUBLE_SIZE;
		m_direct_memory.copyMemory(data, read_from, null, m_buffer+write_to, data.length*m_direct_memory.ARRAY_INT_INDEX_SCALE );		
	}

	/** 
	 * Allocate direct memory buffer and copy matrix contents
	 * @param m_rmatrix
	 * @return
	 */
	public static long copyMatrix(double[][] matrix) {
		long buffer = DirectMemory.allocate(matrix.length * matrix[0].length * DirectMemory.DOUBLE_SIZE);
		long row_length = matrix[0].length;
		for (int i = 0; i < matrix.length ; ++i)
			DirectMemory.writeArray(buffer, i * row_length, matrix[i]);
		return buffer;
	}

	private static Unsafe getUnsafe()
	{
		try {
			Constructor<Unsafe> unsafeConstructor = Unsafe.class.getDeclaredConstructor();
			unsafeConstructor.setAccessible(true);
			Unsafe unsafe = unsafeConstructor.newInstance();
			return unsafe;
		}
		catch (Throwable t){}
		return null;
	}

	public static long readLong(long handle) {
		return m_direct_memory.getLong(handle);
	}

	public static void readArray(long handle, float[] dst) {
		long write_to = Unsafe.ARRAY_FLOAT_BASE_OFFSET;
		long read_from =   handle;
		m_direct_memory.copyMemory(null, read_from, dst, write_to, dst.length*Unsafe.ARRAY_FLOAT_INDEX_SCALE );		
	}
	
	public static void readArray(long handle, int[] dst) {
		long write_to = Unsafe.ARRAY_INT_BASE_OFFSET;
		long read_from =   handle;
		m_direct_memory.copyMemory(null, read_from, dst, write_to, dst.length*Unsafe.ARRAY_INT_INDEX_SCALE );		
	}

	
	public static void readArray(long handle, double[] dst) {
		long write_to = Unsafe.ARRAY_DOUBLE_BASE_OFFSET;
		long read_from =   handle;
		m_direct_memory.copyMemory(null, read_from, dst, write_to, dst.length*Unsafe.ARRAY_DOUBLE_INDEX_SCALE );		
	}

	
	public static void readArray(long handle,  float[] dst, long len) {
		long write_to = Unsafe.ARRAY_FLOAT_BASE_OFFSET;
		long read_from =   handle;
		m_direct_memory.copyMemory(null, read_from, dst, write_to, len*Unsafe.ARRAY_FLOAT_INDEX_SCALE );		
		
	}


}
