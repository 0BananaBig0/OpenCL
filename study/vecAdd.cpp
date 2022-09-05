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

#include <CL/cl.h>

#define DATA_SIZE 10

int main() {
   /* 1. get platform & device information */
   // (1) Check how many OpenCL platforms current system has.
   cl_uint numPlatforms;
   cl_int err = CL_SUCCESS;
   err = clGetPlatformIDs(0, NULL, &numPlatforms);
   if(err != CL_SUCCESS) {
      std::cout << "Your system has 0 OpenCL platform." << std::endl;
      return err;
   }
   // (2) According to the number of OpenCL platforms current system has to
   // mallocate memory so that we can store the information of all available
   // OpenCL platforms.
   std::vector<cl_platform_id> platformIDs(numPlatforms);
   err = clGetPlatformIDs(numPlatforms, &platformIDs[0], NULL);
   if(CL_SUCCESS != err) {
      std::cout << "Your system has no OpenCL platforms for you to use."
                << std::endl;
      return err;
   }
   // (3) Obtain the length of all platforms name and version and store them.
   size_t stringLength;
   std::vector<std::vector<char>> platformNames(numPlatforms);
   std::vector<std::vector<char>> platformVersions(numPlatforms);
   for(uint32_t i = 0; i < numPlatforms; i++) {
      clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, 0, NULL,
                        &stringLength);
      platformNames[i].resize(stringLength);
      clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, stringLength,
                        &platformNames[i][0], NULL);
      clGetPlatformInfo(platformIDs[i], CL_PLATFORM_VERSION, 0, NULL,
                        &stringLength);
      platformVersions[i].resize(stringLength);
      clGetPlatformInfo(platformIDs[i], CL_PLATFORM_VERSION, stringLength,
                        &platformVersions[i][0], NULL);
   }
   // (4) Select platform
   std::vector<std::string> candidatePlatforms = {"Intel(R) CPU",
                                                  "Intel(R) OpenCL", "NVIDIA"};
   uint32_t selectPlatform = 0;
   cl_platform_id selectedPlatformID = NULL;
   for(uint32_t platformIndex = 0; platformIndex < numPlatforms;
       platformIndex++) {
      char *curPlatformName = &platformNames[platformIndex][0];
      for(uint32_t charIndex = 0;
          charIndex < platformNames[platformIndex].size(); charIndex++) {
         uint32_t stringIndex = 0;
         while(curPlatformName[charIndex] ==
               candidatePlatforms[selectPlatform][stringIndex]) {
            charIndex++;
            stringIndex++;
            if(!(charIndex ^ candidatePlatforms[selectPlatform].size())) {
               selectedPlatformID = platformIDs[platformIndex];
               break;
            }
         }
         if(selectedPlatformID != NULL)
            break;
      }
      if(selectedPlatformID != NULL)
         break;
   }
   if(selectedPlatformID == NULL)
      std::cout << "Your system has no such OpenCL platform "
                << candidatePlatforms[selectPlatform] << "." << std::endl;
   // (5)Check how many devices current platform has and store their
   // information.
   cl_uint numDevices = 0;
   cl_device_type deviceType = CL_DEVICE_TYPE_CPU; // select one device type
   err = clGetDeviceIDs(selectedPlatformID, deviceType, 0, NULL, &numDevices);
   if(err != CL_SUCCESS) {
      std::cout << "Current platform " << candidatePlatforms[selectPlatform]
                << " has no supported device." << std::endl;
      return err;
   }
   std::vector<cl_device_id> deviceIDs(numDevices);
   err = clGetDeviceIDs(selectedPlatformID, deviceType, numDevices,
                        &deviceIDs[0], NULL);
   // (6)Obtain device info like obtaining platform info.
   std::vector<std::vector<char>> deviceNames(numDevices);
   for(uint32_t i = 0; i < numDevices; i++) {
      err =
         clGetDeviceInfo(deviceIDs[i], CL_DEVICE_NAME, 0, NULL, &stringLength);
      deviceNames[i].resize(stringLength);
      err = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_NAME, stringLength,
                            &deviceNames[i][0], NULL);
   }
   // (7)Select device like selecting platform.
   std::vector<std::string> candidateDevices = {"Intel(R)", "NVIDIA"};
   uint32_t selectDevice = 0;
   cl_device_id selectedDeviceID = NULL;
   for(uint32_t deviceIndex = 0; deviceIndex < numDevices; deviceIndex++) {
      char *curDeviceName = &deviceNames[deviceIndex][0];
      for(uint32_t charIndex = 0; charIndex < deviceNames[deviceIndex].size();
          charIndex++) {
         uint32_t stringIndex = 0;
         while(curDeviceName[charIndex] ==
               candidateDevices[selectDevice][stringIndex]) {
            charIndex++;
            stringIndex++;
            if(!(charIndex ^ candidateDevices[selectDevice].size())) {
               selectedDeviceID = deviceIDs[deviceIndex];
               break;
            }
         }
         if(selectedDeviceID != NULL)
            break;
      }
      if(selectedDeviceID != NULL)
         break;
   }
   if(selectedDeviceID == NULL)
      std::cout << "Your system has no such OpenCL device "
                << candidateDevices[selectDevice] << "." << std::endl;

   // 2. create context
   // The context is the core of the opencl program and the only channel for the
   // interaction between the host and the device. It is the basis for functions
   // such as memory application maintenance and command queue creation.
   // Different plaforms can apply for different contexts. The memory in
   // different contexts cannot be directly shared. The memory of different
   // devices under the same context is the same and can be accessed by each
   // other.
   cl_context context = NULL;
   cl_context_properties contextProperties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)selectedPlatformID, 0};
   context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                     NULL, NULL, &err);
   if((CL_SUCCESS != err) || (NULL == context)) {
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
   cl_command_queue commandQueue;
   commandQueue = clCreateCommandQueueWithProperties(context, selectedDeviceID, 0, NULL);

   // 4. create program
   // OpenCL C code (written to run on an OpenCL device) is called a program. A
   // program is a collection of functions called kernels, where kernels are
   // units of execution that can be scheduled to run on a device.
   std::ifstream kernelFile("vecAdd.cl", std::ios::in);
   std::ostringstream oss;

   oss << kernelFile.rdbuf();
   std::string srcStdStr = oss.str();
   const char *srcStr = srcStdStr.c_str();
   cl_program program;
   program = clCreateProgramWithSource(context, 1, (const char **)&srcStr,
                                       NULL, NULL);

   // 5. build program
   clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

   // 6. create kernel
   // The final stage to obtain a cl_kernel object that can be used to execute
   // kernels on a device is to extract the kernel from the cl_program.
   // Extracting a kernel from a program is similar to obtaining an exported
   // function from a dynamic library. The name of the kernel that the program
   // exports is used to request it from the compiled program object.
   cl_kernel kernel;
   kernel = clCreateKernel(program, "vecAdd", NULL);

   // 7. set input data && create memory object
   // In order for data to be transferred to a device, it must first be
   // encapsulated as a memory object. OpenCL defines two types of memory
   // objects: buffers and images.
   // Buffers are equivalent to arrays in C, created using malloc(),where data
   // elements are stored contiguously in memory.
   // Whenever a memory object is created, it is valid only within a single
   // context.
   // (1) Declare data in host
   int output[DATA_SIZE];
   int input_x[DATA_SIZE];
   int input_y[DATA_SIZE];
   for(int i = 0; i < DATA_SIZE; i++) {
      input_x[i] = i;
      input_y[i] = 2 * i;
   }
   // (2) Encapsulate them so that we can transfer date into and from devices.
   cl_mem mem_object_x;
   cl_mem mem_object_y;
   cl_mem mem_object_output;
   mem_object_x =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * DATA_SIZE, input_x, NULL);
   mem_object_y =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * DATA_SIZE, input_y, NULL);
   mem_object_output = clCreateBuffer(
      context, CL_MEM_READ_WRITE, sizeof(int) * DATA_SIZE, NULL, NULL);
   clEnqueueWriteBuffer(commandQueue, mem_object_x, false, 0, sizeof(cl_mem), NULL, 0, NULL, NULL);
   clEnqueueWriteBuffer(commandQueue, mem_object_y, false, 0, sizeof(cl_mem), NULL, 0, NULL, NULL);

   // 8. set kernel argument
   // A few more steps are required before the kernel can actually be executed.
   // Unlike calling functions in regular C programs, we cannot simply call a
   // kernel by providing a list of arguments.
   // Executing a kernel requires dispatching it through an enqueue function.
   clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_object_y);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_object_output);
   clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_object_x);

   // 9. send kernel to execute
   // After any required memory objects are transferred to the device and the
   // kernel arguments are set, the kernel is ready to be executed.
   // The clEnqueueNDRangeKernel() call is asynchronous: it will return
   // immediately after the command is enqueued in the command queue and likely
   // before the kernel has even started execution. Either clWaitForEvents() or
   // clFinish() can be used to block execution on the host until the kernel
   // completes.
   size_t globalWorkSize[1] = {DATA_SIZE};
   size_t localWorkSize[1] = {1};
   clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize,
                          localWorkSize, 0, NULL, NULL);

   // 10. read data from output
   // Data contained in host memory is transferred to and from an OpenCL buffer
   // using the commands clEnqueueWriteBuffer() and clEnqueueReadBuffer(),
   // respectively. If a kernel that is dependent on such a buffer is executed
   // on a discrete accelerator device such as a GPU, the buffer may be
   // transferred to the device. The buffer is linked to a context, not a
   // device, so it is the runtime that determines the precise time the data is
   // moved.
   clEnqueueReadBuffer(commandQueue, mem_object_output, CL_TRUE, 0,
                       DATA_SIZE * sizeof(int), output, 0, NULL, NULL);
   for(int i = 0; i < DATA_SIZE; i++) {
      std::cout << output[i] << " ";
   }
   std::cout << std::endl;

   // 11. clean up
   clRetainMemObject(mem_object_x);
   clRetainMemObject(mem_object_y);
   clRetainMemObject(mem_object_output);
   clReleaseCommandQueue(commandQueue);
   clReleaseKernel(kernel);
   clReleaseProgram(program);
   clReleaseContext(context);

   return 0;
}
