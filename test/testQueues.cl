/*************************************************************************
  > File Name: testQueues.cl
  > Author: 16hxliang3
  > Mail: 16hxliang3@stu.edu.cn
  > Created Time: Mon 08 Aug 2022 09:28:13 PM CST
 ************************************************************************/

#include "utility.cl"
static const int limit = 100000000;

__kernel void testQueue1( __global float* output ) {
   unsigned int gid = compute_flattened_global_id();
   static int Test_Queue1 = 0;
   __private float counter = 1;
   for( int i = 0; i < limit; i++ ) {
      counter += i * counter;
   }
   output[gid] = counter;
   if( gid == 1 ) {
      ++Test_Queue1;
      printf( "testQueue1, %d\n", Test_Queue1 );
   }
}

__kernel void testQueue2( __global float* output ) {
   unsigned int gid = compute_flattened_global_id();
   static int Test_Queue2 = 0;
   __private float counter = 1;
   for( int i = 0; i < limit; i++ ) {
      counter += i * counter;
   }
   output[gid] = counter;
   if( gid == 1 ) {
      ++Test_Queue2;
      printf( "testQueue2, %d\n", Test_Queue2 );
   }
}

__kernel void testQueue3( __global float* output ) {
   unsigned int gid = compute_flattened_global_id();
   static int Test_Queue3 = 0;
   __private float counter = 1;
   for( int i = 0; i < limit; i++ ) {
      counter += i * counter;
   }
   output[gid] = counter;
   if( gid == 1 ) {
      ++Test_Queue3;
      printf( "testQueue3, %d\n", Test_Queue3 );
   }
}

__kernel void testQueue4( __global float* output ) {
   unsigned int gid = compute_flattened_global_id();
   static int Test_Queue4 = 0;
   __private float counter = 1;
   for( int i = 0; i < limit; i++ ) {
      counter += i * counter;
   }
   output[gid] = counter;
   if( gid == 1 ) {
      ++Test_Queue4;
      printf( "testQueue4, %d\n", Test_Queue4 );
   }
}

