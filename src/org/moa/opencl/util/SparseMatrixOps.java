package org.moa.opencl.util;



import org.moa.gpu.SparseMatrix;
import org.viennacl.binding.Buffer;
import org.viennacl.binding.Context;
import org.viennacl.binding.DirectMemory;
import org.viennacl.binding.Kernel;

public class SparseMatrixOps extends AbstractUtil {
	
	
	private Context m_context;
	public Kernel m_mult;
	private Kernel m_column_sum;
	private Buffer m_errors;
	private CLogsVarKeyJava m_sorter; 

	public SparseMatrixOps(Context ctx)
	{
		m_context = ctx;
		if (!m_context.hasProgram("csr_matrix_help"))
		{
			m_context.add("csr_matrix_help", generate("double").toString(), "");
		}
		m_mult = m_context.getKernel("csr_matrix_help","vec_mul");
		m_column_sum = m_context.getKernel("csr_matrix_help", "column_sum_2");
		m_errors = new Buffer(ctx, DirectMemory.LONG_SIZE);
		if (m_mult == null)
			throw new RuntimeException("Kernel not found");
		
		//m_sorter = new CLogsVarKeyJava(ctx, true, "unsigned int" , "double"); 
	}
	
	
	public void columnSum(SparseMatrix sp, Buffer result)
	{
		//m_sorter.sortFixedBuffer(sp.getColumnIndices(), sp.getElements(), sp.getRowPostion());


		m_column_sum.set_local_size(0, 256);
		m_column_sum.set_global_size(0, 256 * 40);
		m_column_sum.set_arg(0, sp.getColumnIndices());
		m_column_sum.set_arg(1,  sp.getRowPostion());
		m_column_sum.set_arg(2,  sp.getColumnCount());
		m_column_sum.set_arg(3,  sp.getElements());
		m_column_sum.set_arg(4, result);
		
		m_column_sum.invoke();
	

	}
	
	public void mult(SparseMatrix sp, Buffer vec, Buffer result)
	{
		m_mult.set_local_size(0, 256);
		m_mult.set_global_size(0, 256 * sp.getRowBlockNum());
		m_mult.set_arg(0, sp.getRowJumper());
		m_mult.set_arg(1, sp.getColumnIndices());
		m_mult.set_arg(2,  sp.getRowBlocks());
		m_mult.set_arg(3,  sp.getElements());
		m_mult.set_arg(4,  sp.getRowBlockNum());
		m_mult.set_arg(5,  vec);
		m_mult.set_arg(6,  result);
		m_mult.invoke();
		
	}
	
	
	private String type() {
		return "#define VALUE_TYPE double\n" +
				"#define COND_TYPE long\n";
	}
	
	/** 
	 * CSR matrix mult from viennacl
	 * @param numeric_string
	 * @return
	 */
	public StringBuffer generate(String numeric_string)
	{
		StringBuffer source = new StringBuffer();
		source.append("#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n");
		source.append(type());
		source.append(loadKernel("column_sum.cl"));
		
		  source.append("__kernel void vec_mul( \n");
		  source.append("  __global const unsigned int * row_indices, \n");
		  source.append("  __global const unsigned int * column_indices, \n");
		  source.append("  __global const unsigned int * row_blocks, \n");
		  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
		  source.append("  unsigned int num_blocks, \n");
		  source.append("  __global const "); source.append(numeric_string); source.append(" * x, \n");
		  source.append("  __global "); source.append(numeric_string); source.append(" * result \n");
		  source.append("  ) \n");
		  source.append("{ \n");
		  source.append("  __local "); source.append(numeric_string); source.append(" shared_elements[1024]; \n");

		  source.append("  unsigned int row_start = row_blocks[get_group_id(0)]; \n");
		  source.append("  unsigned int row_stop  = row_blocks[get_group_id(0) + 1]; \n");
		  source.append("  unsigned int rows_to_process = row_stop - row_start; \n");
		  source.append("  unsigned int element_start = row_indices[row_start]; \n");
		  source.append("  unsigned int element_stop = row_indices[row_stop]; \n");

		  source.append("  if (rows_to_process > 4) { \n"); // CSR stream
		      // load to shared buffer:
		  source.append("    for (unsigned int i = element_start + get_local_id(0); i < element_stop; i += get_local_size(0)) \n");
		  source.append("      shared_elements[i - element_start] = elements[i] * x[column_indices[i] ]; \n");

		  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");

		      // use one thread per row to sum:
		  source.append("    for (unsigned int row = row_start + get_local_id(0); row < row_stop; row += get_local_size(0)) { \n");
		  source.append("      "); source.append(numeric_string); source.append(" dot_prod = 0; \n");
		  source.append("      unsigned int thread_row_start = row_indices[row]     - element_start; \n");
		  source.append("      unsigned int thread_row_stop  = row_indices[row + 1] - element_start; \n");
		  source.append("      for (unsigned int i = thread_row_start; i < thread_row_stop; ++i) \n");
		  source.append("        dot_prod += shared_elements[i]; \n");
		  source.append("      result[row ] = dot_prod; \n");
		  source.append("    } \n");
		  source.append("  } \n");

		      // use multiple threads for the summation
		  source.append("  else if (rows_to_process > 1) \n"); // CSR stream with local reduction
		  source.append("  {\n");
		      // load to shared buffer:
		  source.append("    for (unsigned int i = element_start + get_local_id(0); i < element_stop; i += get_local_size(0))\n");
		  source.append("      shared_elements[i - element_start] = elements[i] * x[column_indices[i] ];\n");

		  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");

		    // sum each row separately using a reduction:
		  source.append("    for (unsigned int row = row_start; row < row_stop; ++row)\n");
		  source.append("    {\n");
		  source.append("      unsigned int current_row_start = row_indices[row]     - element_start;\n");
		  source.append("      unsigned int current_row_stop  = row_indices[row + 1] - element_start;\n");
		  source.append("      unsigned int thread_base_id  = current_row_start + get_local_id(0);\n");

		      // sum whatever exceeds the current buffer:
		  source.append("      for (unsigned int j = thread_base_id + get_local_size(0); j < current_row_stop; j += get_local_size(0))\n");
		  source.append("        shared_elements[thread_base_id] += shared_elements[j];\n");

		      // reduction
		  source.append("      for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2)\n");
		  source.append("      {\n");
		  source.append("        barrier(CLK_LOCAL_MEM_FENCE);\n");
		  source.append("        if (get_local_id(0) < stride && thread_base_id < current_row_stop)\n");
		  source.append("          shared_elements[thread_base_id] += (thread_base_id + stride < current_row_stop) ? shared_elements[thread_base_id+stride] : 0;\n");
		  source.append("      }\n");
		  source.append("      "); source.append(numeric_string); source.append(" row_result = 0; \n");
		  source.append("      if (current_row_stop > current_row_start)\n");
		  source.append("        row_result = shared_elements[current_row_start]; \n");
		  source.append("      if (get_local_id(0) == 0)\n");
		  source.append("        result[row ] = row_result;\n");
		  source.append("    }\n");
		  source.append("  }\n");


		  source.append("  else  \n"); // CSR vector for a single row
		  source.append("  { \n");
		      // load and sum to shared buffer:
		  source.append("    shared_elements[get_local_id(0)] = 0; \n");
		  source.append("    for (unsigned int i = element_start + get_local_id(0); i < element_stop; i += get_local_size(0)) \n");
		  source.append("      shared_elements[get_local_id(0)] += elements[i] * x[column_indices[i]]; \n");

		      // reduction to obtain final result
		  source.append("    for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) { \n");
		  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
		  source.append("      if (get_local_id(0) < stride) \n");
		  source.append("        shared_elements[get_local_id(0)] += shared_elements[get_local_id(0) + stride]; \n");
		  source.append("    } \n");

		  source.append("    if (get_local_id(0) == 0) \n");
		  source.append("      result[row_start ] = shared_elements[0]; \n");
		  source.append("  } \n");

		  source.append("} \n");
		  return source;
		}
}
