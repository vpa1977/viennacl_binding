package org.viennacl.binding;

import java.io.File;
import sun.nio.ch.FileChannelImpl;

import java.io.RandomAccessFile;
import java.lang.reflect.Method;
import java.nio.channels.FileChannel;

public class MappedFile {
  
  private static Method map0 = getMethod(FileChannelImpl.class, "map0", int.class, long.class, long.class);
    private static Method unmap0 = getMethod(FileChannelImpl.class, "unmap0", long.class, long.class);

    private long addr;
    private long size;
    
    

    public MappedFile(String name, long size) throws Exception {
         name = "/tmp/" + name;
        size = (size + 0xfffL) & ~0xfffL;

        RandomAccessFile f = new RandomAccessFile(name, "rw");
        FileChannel ch = null;

        try {
            f.setLength(size);
            ch = f.getChannel();
            this.addr = (Long) map0.invoke(ch, 1, 0L, size);
            this.size = size;
        } finally {
            if (ch != null) {
                ch.close();
            }
            f.close();
        }
    }

    public void close() {
        if (addr != 0) {
            try {
                unmap0.invoke(null, addr, size);
            } catch (Exception e) {
                // ignore
            }
            addr = 0;
        }
    }

    public final long getAddr() {
        return addr;
    }

    public final long getSize() {
        return size;
    }
    
    public static Method getMethod(Class cls, String name, Class... params) {
        try {
            Method m = cls.getDeclaredMethod(name, params);
            m.setAccessible(true);
            return m;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }
    
/*
    private long addr;
    private long size;
    
    

    public MappedFile(String name, long size) throws Exception {
        name = "/tmp/"+ name;
        //if (!new File(name).exists()) 
        //    new File(name).createNewFile();
        size = (size + 0xfffL) & ~0xfffL;
        addr = nativeAttach(name, size);
    }
    
    private native long nativeAttach(String key, long size);
    private native void nativeDetach(long addr);

    public void close() {
        if (addr != 0) {
            nativeDetach(addr);
            addr = 0;
        }
    }

    public final long getAddr() {
        return addr;
    }

    public final long getSize() {
        return size;
    }
  */  
   
}