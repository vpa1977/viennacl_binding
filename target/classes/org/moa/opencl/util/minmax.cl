/*
 computes min/max values in the range
*/
/***************************************************************************
*  Based on bolt::cl::max_element function:
*
*   © 2012,2014 Advanced Micro Devices, Inc. All rights reserved.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/
#define GROUP_SIZE 256
#pragma OPENCL EXTENSION cl_amd_printf : enable
#define REDUCE_STEP_MIN_MAX(length, index, width)\
   if (index<width && (index + width)<length){\
      VALUE_TYPE mine = scratch_max[index];\
      VALUE_TYPE other = scratch_max[(index + width)];\
      if (other > mine)\
      {\
      	scratch_max[index] = other;\
      }\
      mine = scratch_min[index];\
      other= scratch_min[(index + width)];\
      if (other < mine)\
      {\
      	scratch_min[index] = other;\
      }\
   }\
   barrier(CLK_LOCAL_MEM_FENCE);


__kernel void min_max_update_kernel(
	int class_attribute,
	int attribute_count, /* number of attributes */
   __global VALUE_TYPE *input, /* attribute vector */
	 __global VALUE_TYPE* result_min, /* min values per attribute */
	 __global VALUE_TYPE* result_max /* max value per attribute */
)
{
	int id = get_global_id(0);
	if (id < attribute_count)
	{
		if (result_min[id] > input[id])
			result_min[id] = input[id];
		if (result_max[id] < input[id])
			result_max[id] = input[id];
	}
}

/**
	scan min/max value.
	1 workgroup per attribute
	output 1 entry per workgroup
*/
__kernel void min_max_kernel(
	int class_attribute,
	int stride, /* number of attributes */
	int length, /* number of instances */
   __global VALUE_TYPE *input, /* attribute vector */
	 __global VALUE_TYPE* result_min, /* min values per attribute */
	 __global VALUE_TYPE* result_max /* max value per attribute */
){

  int offset = get_group_id(0);
  int local_index = get_local_id(0);
  __local VALUE_TYPE scratch_max[GROUP_SIZE], scratch_min[GROUP_SIZE];
  int gx = get_local_id(0);
  int gloId = gx;


  VALUE_TYPE accumulator_min, accumulator_max;
  if (gloId<length){
     accumulator_max = input[offset + gx*stride];
     accumulator_min = accumulator_max;
     gx = gx + get_local_size(0);
  }

  for (; gx<length; gx += get_local_size(0)){
     VALUE_TYPE element = input[offset + gx*stride];
     if (element < accumulator_min)
     {
     	accumulator_min = element;
     }
     if (element > accumulator_max)
     {
     	accumulator_max = element;
     }
     barrier(CLK_LOCAL_MEM_FENCE);
  }

  scratch_max[local_index]  = accumulator_max;
  scratch_min[local_index]  = accumulator_min;

  barrier(CLK_LOCAL_MEM_FENCE);
  int tail = length;
  REDUCE_STEP_MIN_MAX( tail, local_index, 128);
  REDUCE_STEP_MIN_MAX( tail, local_index, 64);
  REDUCE_STEP_MIN_MAX( tail, local_index, 32);
  REDUCE_STEP_MIN_MAX( tail, local_index, 16);
  REDUCE_STEP_MIN_MAX( tail, local_index, 8);
  REDUCE_STEP_MIN_MAX( tail, local_index, 4);
  REDUCE_STEP_MIN_MAX( tail, local_index, 2);
  REDUCE_STEP_MIN_MAX( tail, local_index, 1);

  if (local_index==0){
     result_max[ get_group_id(0)]  = scratch_max[0];
     result_min[ get_group_id(0)]  = scratch_min[0];
  }

  return;
}
