////////////////////////////////////////////////////////////////////////////////
///                    File Name: coarse_grained_svm.cpp                     ///
///                          Author: Huaxiao Liang                           ///
///                         Mail: 1184903633@qq.com                          ///
///                         02/18/2025-Tue-12:53:48                          ///
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

   // Allocate SVM buffer
   int* svm_ptr = static_cast< int* >(
      clSVMAlloc( context, CL_MEM_READ_WRITE, data_size * sizeof( int ), 0 ) );
   if( !svm_ptr ) {
      std::cerr << "Failed to allocate SVM buffer" << std::endl;
      exit( 1 );
   }

   // Enqueue SVM memory to device
   err = clEnqueueSVMMap( queue,
                          CL_TRUE,
                          CL_MAP_WRITE,
                          svm_ptr,
                          data_size * sizeof( int ),
                          0,
                          nullptr,
                          nullptr );
   CHECK_ERR( err, "Failed to map SVM memory" );

   // Initialize SVM memory, must after mapping.
   for( int i = 0; i < data_size; i++ ) {
      svm_ptr[i] = 2 * i;
   }

   // Unmap before running kernel
   err = clEnqueueSVMUnmap( queue, svm_ptr, 0, nullptr, nullptr );
   CHECK_ERR( err, "Failed to unmap SVM memory" );

   // Compile and set kernel
   cl_program program
      = clCreateProgramWithSource( context, 1, &kernelSource, nullptr, &err );
   CHECK_ERR( err, "Failed to create program" );
   clBuildProgram( program, 1, &device, nullptr, nullptr, nullptr );
   cl_kernel kernel = clCreateKernel( program, "add_one", &err );
   CHECK_ERR( err, "Failed to create kernel" );

   clSetKernelArgSVMPointer( kernel, 0, svm_ptr );

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

   // Map to read results
   err = clEnqueueSVMMap( queue,
                          CL_TRUE,
                          CL_MAP_READ,
                          svm_ptr,
                          data_size * sizeof( int ),
                          0,
                          nullptr,
                          nullptr );
   CHECK_ERR( err, "Failed to map SVM memory for reading" );

   // Print some results
   std::cout << "First 10 elements: ";
   for( int i = 0; i < 10; i++ ) {
      std::cout << svm_ptr[i] << " ";
   }
   std::cout << std::endl;

   // Unmap SVM memory
   clEnqueueSVMUnmap( queue, svm_ptr, 0, nullptr, nullptr );

   // Cleanup
   clReleaseKernel( kernel );
   clReleaseProgram( program );
   clSVMFree( context, svm_ptr );
   clReleaseCommandQueue( queue );
   clReleaseContext( context );

   return 0;
}

