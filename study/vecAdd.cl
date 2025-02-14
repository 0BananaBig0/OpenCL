/*************************************************************************
  > File Name: vecAdd.cl
  > Author: 16hxliang3
  > Mail: 16hxliang3@stu.edu.cn
  > Created Time: Mon 08 Aug 2022 09:28:13 PM CST
 ************************************************************************/

#include "utility.cl"

__kernel void vecAdd( __global const int* input_x,
                      __global const int* input_y,
                      __global int* output ) {
   unsigned int gid = compute_flattened_global_id();
   // printf( "%d\n", gid );
   // into
   output[gid] = input_x[gid] + input_y[gid];
}

