/*************************************************************************
  > File Name: vecAdd.cl
  > Author: 16hxliang3
  > Mail: 16hxliang3@stu.edu.cn
  > Created Time: Mon 08 Aug 2022 09:28:13 PM CST
 ************************************************************************/

unsigned compute_flattened_global_id() {
   unsigned flatten_id = get_global_id( 0 );
   unsigned multiplicand = get_global_size( 0 );
   unsigned work_dims = get_work_dim();   // number of dimensions in use
   for( unsigned dim = 1; dim < work_dims; dim++ ) {
      // flatten_id += get_global_id( dim ) * multiplicand;
      flatten_id
         = mad24( (unsigned)get_global_id( dim ), multiplicand, flatten_id );
      multiplicand = multiplicand * get_global_size( dim );
   }
   return flatten_id;
}

__kernel void vecAdd( __global const int* input_x,
                      __global const int* input_y,
                      __global int* output ) {
   unsigned int gid = compute_flattened_global_id();
   // printf( "%d\n", gid );
   // into
   output[gid] = input_x[gid] + input_y[gid];
}

__kernel void vecMul( __global const int* input_z,
                      __global const int* input_y,
                      __global int* output ) {
   unsigned int gid = compute_flattened_global_id();
   // printf( "%d\n", gid );
   // into
   output[gid] = input_z[gid] * input_y[gid];
}

