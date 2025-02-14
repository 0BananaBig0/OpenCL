////////////////////////////////////////////////////////////////////////////////
///                           File Name: vecMul.cl                           ///
///                          Author: Huaxiao Liang                           ///
///                         Mail: 1184903633@qq.com                          ///
///                         02/14/2025-Fri-10:00:16                          ///
////////////////////////////////////////////////////////////////////////////////

#include "utility.cl"

__kernel void vecMul( __global const int* input_z,
                      __global const int* input_y,
                      __global int* output ) {
   unsigned int gid = compute_flattened_global_id();
   // printf( "%d\n", gid );
   // into
   output[gid] = input_z[gid] * input_y[gid];
}

