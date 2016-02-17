/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test.org.viennacl.binding.hsa;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import org.junit.Assert;
import org.junit.Test;
import org.moa.opencl.util.CLogsVarKeyJava;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

/**
 *
 * @author bsp
 */
public class KernelParameterTest {
  static 
  {
    System.loadLibrary("viennacl-java-binding");
  }
  @Test
  public void testIntArray()
  {
    Context ctx = new Context(Context.Memory.HSA_MEMORY, null);
    String kernel = "__kernel void upd_int_array(__global uint* int_arr){ int_arr[get_local_id(0)] = get_local_id(0);}";
    ctx.add("newprogram", kernel);
    Kernel sample = ctx.getKernel("newprogram", "upd_int_array");
    int[] test = new int[512];
    Buffer buf = new Buffer(ctx, 512 * DirectMemory.INT_SIZE);
    buf.mapBuffer(Buffer.WRITE);
    buf.writeArray(0, test);
    buf.commitBuffer();
    
    sample.set_global_size(0, 64);
    sample.set_arg(0, buf);
    sample.invoke();
    
    buf.mapBuffer(Buffer.READ);
    buf.readArray(0, test);
    buf.commitBuffer();
    Assert.assertEquals(test[63], 63);
    Assert.assertEquals(test[64], 0);
  }
  
  @Test
  public void testIntArrayWithParam()
  {
    Context ctx = new Context(Context.Memory.HSA_MEMORY, null);
    String kernel = "__kernel void upd_int_array(__global uint* int_arr, const int param){ int_arr[get_local_id(0)] = param;}";
    ctx.add("newprogram", kernel);
    Kernel sample = ctx.getKernel("newprogram", "upd_int_array");
    int[] test = new int[512];
    Buffer buf = new Buffer(ctx, 512 * DirectMemory.INT_SIZE);
    buf.mapBuffer(Buffer.WRITE);
    buf.writeArray(0, test);
    buf.commitBuffer();
    
    sample.set_global_size(0, 64);
    sample.set_arg(0, buf);
    sample.set_arg(1, 1337);
    sample.invoke();
    
    buf.mapBuffer(Buffer.READ);
    buf.readArray(0, test);
    buf.commitBuffer();
    Assert.assertEquals(test[63], 1337);
    Assert.assertEquals(test[64], 0);
  }

  @Test
  public void testIntArrayWithParam1() throws Throwable
  {
    Context ctx = new Context(Context.Memory.HSA_MEMORY, null);
    CLogsVarKeyJava vk = new CLogsVarKeyJava(ctx, false);
    
    ctx.add("newprogram", vk.program);
    Kernel sample = ctx.getKernel("newprogram", "radixsortReduce");
    int[] test = new int[512];
    Buffer buf = new Buffer(ctx, 512 * DirectMemory.INT_SIZE);
    buf.mapBuffer(Buffer.WRITE);
    buf.writeArray(0, test);
    buf.commitBuffer();
    
    vk.sortFixedBuffer(buf, buf, 512);
    
    sample.set_global_size(0, 3 * 128);
    sample.set_local_size(0, 128);
    sample.set_arg(0, buf);
    sample.set_arg(1, buf);
    
    sample.set_arg(2, 1337);
    sample.set_arg(3, 1337);
    sample.set_arg(4, 1337);
    
    sample.set_arg(5, buf);
    
    sample.invoke();
    
    buf.mapBuffer(Buffer.READ);
    buf.readArray(0, test);
    buf.commitBuffer();
    Assert.assertEquals(test[63], 1337);
    Assert.assertEquals(test[64], 0);
  }
  
  
  public static void main(String[] args) throws Throwable
  {
    new KernelParameterTest().testIntArrayWithParam1();
  }
}
