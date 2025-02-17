/*************************************************************************
  > File Name: vecAdd.cpp
  > Author: 16hxliang3
  > Mail: 16hxliang3@stu.edu.cn
  > Created Time: Mon 08 Aug 2022 09:16:47 PM CST
 ************************************************************************/

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stddef.h>
#include <vector>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#define DATA_SIZE 512

int main() {
   /* 1. get platform & device information */
   // (1) Check how many OpenCL platforms current system has.
   cl_uint num_platforms;
   cl_int err = CL_SUCCESS;
   err = clGetPlatformIDs( 0, nullptr, &num_platforms );
   if( err != CL_SUCCESS ) {
      std::cout << "Your system has 0 OpenCL platform." << std::endl;
      return err;
   }

   // (2) According to the number of OpenCL platforms current system has to
   // mallocate memory so that we can store the information of all available
   // OpenCL platforms.
   std::vector< cl_platform_id > platform_ids( num_platforms );
   err = clGetPlatformIDs( num_platforms, &platform_ids[0], nullptr );
   if( CL_SUCCESS != err ) {
      std::cout << "Your system has no OpenCL platforms for you to use."
                << std::endl;
      return err;
   }

   // (3) Obtain the length of all platforms name and version and store them.
   size_t string_length;
   std::vector< std::string > platform_names( num_platforms );
   std::vector< std::string > platform_versions( num_platforms );
   for( uint32_t i = 0; i < num_platforms; i++ ) {
      clGetPlatformInfo( platform_ids[i],
                         CL_PLATFORM_NAME,
                         0,
                         nullptr,
                         &string_length );
      platform_names[i].resize( string_length );
      clGetPlatformInfo( platform_ids[i],
                         CL_PLATFORM_NAME,
                         string_length,
                         &platform_names[i][0],
                         nullptr );

      clGetPlatformInfo( platform_ids[i],
                         CL_PLATFORM_VERSION,
                         0,
                         nullptr,
                         &string_length );
      platform_versions[i].resize( string_length );
      clGetPlatformInfo( platform_ids[i],
                         CL_PLATFORM_VERSION,
                         string_length,
                         &platform_versions[i][0],
                         nullptr );
   }

   // (4) Select platform
   std::vector< std::string > candidate_platforms = { "Intel(R)", "NVIDIA" };
   uint32_t select_platform = 1;
   cl_platform_id selected_platform_id = nullptr;
   for( uint32_t platform_index = 0; platform_index < num_platforms;
        platform_index++ ) {
      if( platform_names[platform_index].find(
             candidate_platforms[select_platform] )
          != std::string::npos ) {
         selected_platform_id = platform_ids[platform_index];
         break;
      }
   }
   if( selected_platform_id == nullptr ) {
      std::cout << "Your system has no such OpenCL platform "
                << candidate_platforms[select_platform] << "." << std::endl;
   }

   // (5)Check how many devices current platform has and store their
   // information.
   cl_uint num_devices = 0;
   // Select one device type.
   cl_device_type device_type = CL_DEVICE_TYPE_GPU;
   // Get the number of devices.
   err = clGetDeviceIDs( selected_platform_id,
                         device_type,
                         0,
                         nullptr,
                         &num_devices );
   if( err != CL_SUCCESS ) {
      std::cout << "Current platform " << candidate_platforms[select_platform]
                << " has no supported device." << std::endl;
      return err;
   }
   // Get all devices' ids.
   std::vector< cl_device_id > device_ids( num_devices );
   err = clGetDeviceIDs( selected_platform_id,
                         device_type,
                         num_devices,
                         &device_ids[0],
                         nullptr );

   // (6)Obtain device info like obtaining platform info.
   std::vector< std::string > device_names( num_devices );
   for( uint32_t i = 0; i < num_devices; i++ ) {
      err = clGetDeviceInfo( device_ids[i],
                             CL_DEVICE_NAME,
                             0,
                             nullptr,
                             &string_length );
      device_names[i].resize( string_length );
      err = clGetDeviceInfo( device_ids[i],
                             CL_DEVICE_NAME,
                             string_length,
                             &device_names[i][0],
                             nullptr );
   }

   // (7)Select device like selecting platform.
   std::vector< std::string > candidate_devices = { "Intel(R)", "NVIDIA" };
   uint32_t select_device = 1;
   cl_device_id selected_device_id = nullptr;
   for( uint32_t device_index = 0; device_index < num_devices;
        device_index++ ) {
      if( device_names[device_index].find( candidate_devices[select_device] )
          != std::string::npos ) {
         selected_device_id = device_ids[device_index];
         break;
      }
   }
   if( selected_device_id == nullptr ) {
      std::cout << "Your system has no such OpenCL device "
                << candidate_devices[select_device] << "." << std::endl;
   }

   // 2. create context
   // The context is the core of the opencl program and the only channel for the
   // interaction between the host and the device. It is the basis for functions
   // such as memory application maintenance and command queue creation.
   // Different plaforms can apply for different contexts. The memory in
   // different contexts cannot be directly shared. The memory of different
   // devices under the same context is the same and can be accessed by each
   // other.
   cl_context context = nullptr;
   cl_context_properties context_properties[]
      = { CL_CONTEXT_PLATFORM,
          reinterpret_cast< cl_context_properties >( selected_platform_id ),
          0 };
   context = clCreateContextFromType( context_properties,
                                      CL_DEVICE_TYPE_GPU,
                                      nullptr,
                                      nullptr,
                                      &err );
   if( ( CL_SUCCESS != err ) || ( nullptr == context ) ) {
      std::cout
         << "Couldn't create a context, clCreateContextFromType() returned "
         << err << std::endl;
      return err;
   }

   // 3. create command queue
   // Communication with a device occurs by submitting commands to a command
   // queue. The command queue is the mechanism that the host uses to request
   // action by the device. Once the host decides which devices to work with and
   // a context is created, one command queue needs to be created per device
   // (i.e., each command queue is associated with only one device). Whenever
   // the host needs an action to be performed by a device, it will submit
   // commands to the proper command queue. Any API that specifies hostdevice
   // interaction will always begin with clEnqueue and require a command queue
   // as a parameter
   auto command_queue1 = clCreateCommandQueueWithProperties( context,
                                                             selected_device_id,
                                                             nullptr,
                                                             nullptr );
   auto command_queue2 = clCreateCommandQueueWithProperties( context,
                                                             selected_device_id,
                                                             nullptr,
                                                             nullptr );
   auto command_queue3 = clCreateCommandQueueWithProperties( context,
                                                             selected_device_id,
                                                             nullptr,
                                                             nullptr );
   auto command_queue4 = clCreateCommandQueueWithProperties( context,
                                                             selected_device_id,
                                                             nullptr,
                                                             nullptr );

   // 4. create program
   // OpenCL C code (written to run on an OpenCL device) is called a program. A
   // program is a collection of functions called kernels, where kernels are
   // units of execution that can be scheduled to run on a device.
   std::ifstream kernel_file( "testQueues.cl", std::ios::in );
   std::ostringstream oss;

   oss << kernel_file.rdbuf();
   std::string src_std_str = oss.str();
   const char* src_str = src_std_str.c_str();
   auto program
      = clCreateProgramWithSource( context, 1, &src_str, nullptr, nullptr );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to create program program.\n";
   }

   // 5. build program
   err = clBuildProgram( program,
                         1,
                         &selected_device_id,
                         "-I.",
                         nullptr,
                         nullptr );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to build program program.\n";
   }

   // 6. create kernel
   // The final stage to obtain a cl_kernel object that can be used to execute
   // kernels on a device is to extract the kernel from the cl_program.
   // Extracting a kernel from a program is similar to obtaining an exported
   // function from a dynamic library. The name of the kernel that the program
   // exports is used to request it from the compiled program object.
   auto test_queue1 = clCreateKernel( program, "testQueue1", &err );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to create kernel testQueue1.\n";
   }
   auto test_queue2 = clCreateKernel( program, "testQueue2", &err );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to create kernel testQueue2.\n";
   }
   auto test_queue3 = clCreateKernel( program, "testQueue3", &err );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to create kernel testQueue3.\n";
   }
   auto test_queue4 = clCreateKernel( program, "testQueue4", &err );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to create kernel testQueue4.\n";
   }

   // 7. set input data && create memory object
   // In order for data to be transferred to a device, it must first be
   // encapsulated as a memory object. OpenCL defines two types of memory
   // objects: buffers and images.
   // Buffers are equivalent to arrays in C, created using malloc(),where data
   // elements are stored contiguously in memory.
   // Whenever a memory object is created, it is valid only within a single
   // context.
   // (1) Declare data in host
   float output[DATA_SIZE];
   // (2) Encapsulate them so that we can transfer date into and from devices.
   auto mem_object_output = clCreateBuffer( context,
                                            CL_MEM_READ_WRITE,
                                            sizeof( float ) * DATA_SIZE,
                                            nullptr,
                                            &err );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to create buffer mem_object_output.\n";
   }

   // 8. set kernel argument
   // A few more steps are required before the kernel can actually be executed.
   // Unlike calling functions in regular C programs, we cannot simply call a
   // kernel by providing a list of arguments.
   // Executing a kernel requires dispatching it through an enqueue function.
   err = clSetKernelArg( test_queue1,
                         0,
                         sizeof( cl_mem ),
                         static_cast< const void* >( &mem_object_output ) );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to set arg 0 for test_queue1.\n";
   }
   err = clSetKernelArg( test_queue2,
                         0,
                         sizeof( cl_mem ),
                         static_cast< const void* >( &mem_object_output ) );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to set arg 0 for test_queue2.\n";
   }
   err = clSetKernelArg( test_queue3,
                         0,
                         sizeof( cl_mem ),
                         static_cast< const void* >( &mem_object_output ) );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to set arg 0 for test_queue3.\n";
   }
   err = clSetKernelArg( test_queue4,
                         0,
                         sizeof( cl_mem ),
                         static_cast< const void* >( &mem_object_output ) );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to set arg 0 for test_queue4.\n";
   }

   // 9. send kernel to execute
   // After any required memory objects are transferred to the device and the
   // kernel arguments are set, the kernel is ready to be executed.
   // The clEnqueueNDRangeKernel() call is asynchronous: it will return
   // immediately after the command is enqueued in the command queue and likely
   // before the kernel has even started execution. Either clWaitForEvents() or
   // clFinish() can be used to block execution on the host until the kernel
   // completes.
   // global_work_size must be a multiple of local_work_size in each dimension
   // when using clEnqueueNDRangeKernel. If they are not, OpenCL will return an
   // error or produce unexpected behavior.
   constexpr size_t work_dim = 3;
   size_t global_work_size[work_dim] = { DATA_SIZE, 1, 1 };
   size_t local_work_size[work_dim] = { 256, 1, 1 };
   err = clEnqueueNDRangeKernel( command_queue1,
                                 test_queue1,
                                 work_dim,
                                 nullptr,
                                 global_work_size,
                                 local_work_size,
                                 0,
                                 nullptr,
                                 nullptr );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to enqueue kernel testQueue1.\n";
   }
   err = clEnqueueNDRangeKernel( command_queue2,
                                 test_queue2,
                                 work_dim,
                                 nullptr,
                                 global_work_size,
                                 local_work_size,
                                 0,
                                 nullptr,
                                 nullptr );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to enqueue kernel testQueue2.\n";
   }
   err = clEnqueueNDRangeKernel( command_queue3,
                                 test_queue3,
                                 work_dim,
                                 nullptr,
                                 global_work_size,
                                 local_work_size,
                                 0,
                                 nullptr,
                                 nullptr );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to enqueue kernel testQueue3.\n";
   }
   err = clEnqueueNDRangeKernel( command_queue4,
                                 test_queue4,
                                 work_dim,
                                 nullptr,
                                 global_work_size,
                                 local_work_size,
                                 0,
                                 nullptr,
                                 nullptr );
   if( err != CL_SUCCESS ) {
      std::cout << "Fail to enqueue kernel testQueue4.\n";
   }

   err = clEnqueueReadBuffer( command_queue1,
                              mem_object_output,
                              CL_TRUE,
                              0,
                              DATA_SIZE * sizeof( int ),
                              &output[0],
                              0,
                              nullptr,
                              nullptr );

   // 12. clean up
   clReleaseCommandQueue( command_queue1 );
   clReleaseCommandQueue( command_queue2 );
   clReleaseCommandQueue( command_queue3 );
   clReleaseCommandQueue( command_queue4 );
   clReleaseMemObject( mem_object_output );
   clReleaseKernel( test_queue1 );
   clReleaseKernel( test_queue2 );
   clReleaseKernel( test_queue3 );
   clReleaseKernel( test_queue4 );
   clReleaseProgram( program );
   clReleaseContext( context );

   return 0;
}

