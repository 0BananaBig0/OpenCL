////////////////////////////////////////////////////////////////////////////////
///                        File Name: sub_buffer.cpp                         ///
///                          Author: Huaxiao Liang                           ///
///                         Mail: 1184903633@qq.com                          ///
///                         02/18/2025-Tue-15:27:22                          ///
////////////////////////////////////////////////////////////////////////////////

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
   const int data_size = 1024;
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

   // Create parent buffer
   cl_mem parent_buffer = clCreateBuffer( context,
                                          CL_MEM_READ_WRITE,
                                          data_size * sizeof( int ),
                                          nullptr,
                                          &err );
   CHECK_ERR( err, "Failed to create parent buffer" );

   // Create a sub-buffer that represents half of the parent buffer
   // Modify the range of the region, you can know how it works.
   cl_buffer_region region = { 0, data_size / 2 * sizeof( int ) };
   cl_mem sub_buffer = clCreateSubBuffer( parent_buffer,
                                          CL_MEM_READ_WRITE,
                                          CL_BUFFER_CREATE_TYPE_REGION,
                                          &region,
                                          &err );
   CHECK_ERR( err, "Failed to create sub-buffer" );

   // Map parent buffer to host and initialize data
   int* host_ptr
      = static_cast< int* >( clEnqueueMapBuffer( queue,
                                                 parent_buffer,
                                                 CL_TRUE,
                                                 CL_MAP_WRITE,
                                                 0,
                                                 data_size * sizeof( int ),
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 &err ) );
   CHECK_ERR( err, "Failed to map parent buffer" );

   for( int i = 0; i < data_size; i++ ) {
      host_ptr[i] = i * 4;
   }

   clEnqueueUnmapMemObject( queue,
                            parent_buffer,
                            host_ptr,
                            0,
                            nullptr,
                            nullptr );

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
                   static_cast< const void* >( &sub_buffer ) );

   // Execute kernel on sub-buffer
   size_t global_size = data_size / 2;   // Only modifying half
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

   clFinish( queue );

   // Map buffer for reading results
   host_ptr
      = static_cast< int* >( clEnqueueMapBuffer( queue,
                                                 parent_buffer,
                                                 CL_TRUE,
                                                 CL_MAP_READ,
                                                 0,
                                                 data_size * sizeof( int ),
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 &err ) );
   CHECK_ERR( err, "Failed to map parent buffer for reading" );

   // Print first 10 elements
   std::cout << "First 10 elements: ";
   for( int i = 0; i < 10; i++ ) {
      std::cout << host_ptr[i] << " ";
   }
   std::cout << std::endl;

   // Unmap buffer
   clEnqueueUnmapMemObject( queue,
                            parent_buffer,
                            host_ptr,
                            0,
                            nullptr,
                            nullptr );

   // Cleanup
   clReleaseKernel( kernel );
   clReleaseProgram( program );
   clReleaseMemObject( sub_buffer );
   clReleaseMemObject( parent_buffer );
   clReleaseCommandQueue( queue );
   clReleaseContext( context );

   return 0;
}

