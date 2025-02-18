/*************************************************************************
  > File Name: map.cpp
  > Author: 16hxliang3
  > Mail: 16hxliang3@stu.edu.cn
  > Created Time: Mon 08 Aug 2022 09:16:47 PM CST
 ************************************************************************/

#include <iostream>
#include <stddef.h>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#include <iostream>

#define CHECK_ERR( err, msg )                                  \
   if( err != CL_SUCCESS ) {                                   \
      std::cerr << msg << " Error Code: " << err << std::endl; \
      exit( 1 );                                               \
   }

const char* kernelSource = R"(
__kernel void add_one(__global int* data) {
    int i = get_global_id(0);
    data[i] += 3;
}
)";

int main() {
   constexpr int data_size = 1024;
   cl_int err;

   // Get platform and device
   cl_platform_id platform;
   cl_device_id device;
   clGetPlatformIDs( 1, &platform, nullptr );
   clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr );

   // Create context and queue
   cl_context context
      = clCreateContext( nullptr, 1, &device, nullptr, nullptr, &err );
   CHECK_ERR( err, "Failed to create context" );

   cl_command_queue queue
      = clCreateCommandQueueWithProperties( context, device, nullptr, &err );
   CHECK_ERR( err, "Failed to create command queue" );

   // Create buffer
   cl_mem buffer = clCreateBuffer( context,
                                   CL_MEM_READ_WRITE,
                                   data_size * sizeof( int ),
                                   nullptr,
                                   &err );
   CHECK_ERR( err, "Failed to create buffer" );

   // Map buffer (Coarse-Grained Memory)
   // Mapped memory access is for host use
   int* mapped_ptr
      = static_cast< int* >( clEnqueueMapBuffer( queue,
                                                 buffer,
                                                 CL_TRUE,
                                                 CL_MAP_WRITE,
                                                 0,
                                                 data_size * sizeof( int ),
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 &err ) );
   CHECK_ERR( err, "Failed to map buffer" );

   // Initialize data, must after mapping
   for( int i = 0; i < data_size; i++ ) {
      mapped_ptr[i] = 2 * i;
   }

   // Unmap buffer, must unmap the buffer before kernel execution
   // Unmapping transfers ownership back to OpenCL
   err = clEnqueueUnmapMemObject( queue,
                                  buffer,
                                  mapped_ptr,
                                  0,
                                  nullptr,
                                  nullptr );
   CHECK_ERR( err, "Failed to unmap buffer" );

   // Compile and set kernel
   cl_program program
      = clCreateProgramWithSource( context, 1, &kernelSource, nullptr, &err );
   CHECK_ERR( err, "Failed to create program" );
   clBuildProgram( program, 1, &device, nullptr, nullptr, nullptr );
   cl_kernel kernel = clCreateKernel( program, "add_one", &err );
   CHECK_ERR( err, "Failed to create kernel" );

   clSetKernelArg( kernel,
                   0,
                   sizeof( cl_mem ),
                   static_cast< const void* >( &buffer ) );

   // Execute kernel
   size_t global_size = data_size;
   err = clEnqueueNDRangeKernel( queue,
                                 kernel,
                                 1,
                                 nullptr,
                                 &global_size,
                                 nullptr,
                                 0,
                                 nullptr,
                                 nullptr );
   CHECK_ERR( err, "Failed to enqueue kernel" );

   // Map buffer to read results
   mapped_ptr
      = static_cast< int* >( clEnqueueMapBuffer( queue,
                                                 buffer,
                                                 CL_TRUE,
                                                 CL_MAP_READ,
                                                 0,
                                                 data_size * sizeof( int ),
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 &err ) );
   CHECK_ERR( err, "Failed to map buffer for reading" );

   // Print some results
   std::cout << "First 10 elements: ";
   for( int i = 0; i < 10; i++ ) {
      std::cout << mapped_ptr[i] << " ";
   }
   std::cout << std::endl;

   // Unmap buffer
   clEnqueueUnmapMemObject( queue, buffer, mapped_ptr, 0, nullptr, nullptr );

   // Cleanup
   clReleaseKernel( kernel );
   clReleaseProgram( program );
   clReleaseMemObject( buffer );
   clReleaseCommandQueue( queue );
   clReleaseContext( context );

   return 0;
}

