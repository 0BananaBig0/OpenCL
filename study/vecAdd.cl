/*************************************************************************
  > File Name: vecAdd.cl
  > Author: 16hxliang3
  > Mail: 16hxliang3@stu.edu.cn
  > Created Time: Mon 08 Aug 2022 09:28:13 PM CST
 ************************************************************************/

__kernel void vector_add(__global const float *input_x,
                         __global const float *input_y,
                         __global float *output) {
  int gid = get_global_id(0);

  output[gid] = input_x[gid] + input_y[gid];
}
