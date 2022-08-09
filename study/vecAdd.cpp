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

const int DATA_SIZE = 10;

int main(void) {
  /* 1. get platform & device information */
  // (1) check how many OpenCL platforms current system has.
  cl_uint numPlatforms;
  cl_int err = CL_SUCCESS;
  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (err != CL_SUCCESS) {
    std::cout << "Your system has 0 OpenCL platform." << std::endl;
    return 0;
  }
  // (2) According to the number of OpenCL platforms current system has to
  // mallocate memory so that we can store the information of all available
  // OpenCL platforms.
  std::vector<cl_platform_id> platformIDs(numPlatforms);
  err = clGetPlatformIDs(numPlatforms, &platformIDs[0], NULL);
  if (CL_SUCCESS != err) {
    std::cout << "Your system has no OpenCL platforms for you to use."
              << std::endl;
    return 0;
  }
  // (3) Obtain the length of all platforms name and version and store them.
  size_t stringLength;
  std::vector<std::vector<char>> platformNames(numPlatforms);
  std::vector<std::vector<char>> platformVersions(numPlatforms);
  for (uint32_t i = 0; i < numPlatforms; i++) {
    clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, 0, NULL, &stringLength);
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
  for (uint32_t platformIndex = 0; platformIndex < numPlatforms;
       platformIndex++) {
    char *curPlatformName = &platformNames[platformIndex][0];
    for (uint32_t charIndex = 0;
         charIndex < platformNames[platformIndex].size(); charIndex++) {
      uint32_t stringIndex = 0;
      while (curPlatformName[charIndex] ==
             candidatePlatforms[selectPlatform][stringIndex]) {
        charIndex++;
        stringIndex++;
        if (!(charIndex ^ candidatePlatforms[selectPlatform].size())) {
          selectedPlatformID = platformIDs[platformIndex];
          break;
        }
      }
      if (selectedPlatformID != NULL)
        break;
    }
    if (selectedPlatformID != NULL)
      break;
  }
  if (selectedPlatformID == NULL)
    std::cout << "Your system has no such OpenCL platform "
              << candidatePlatforms[selectPlatform] << "." << std::endl;
  // (5)Check how many devices current platform has and store their information.
  cl_uint numDevices = 0;
  cl_device_type deviceType = CL_DEVICE_TYPE_CPU; // select one device type
  err = clGetDeviceIDs(selectedPlatformID, deviceType, 0, NULL, &numDevices);
  if (err != CL_SUCCESS) {
    std::cout << "Current platform " << candidatePlatforms[selectPlatform]
              << " has no supported device." << std::endl;
    return 0;
  }
  std::vector<cl_device_id> deviceIDs(numDevices);
  err = clGetDeviceIDs(selectedPlatformID, deviceType, numDevices,
                       &deviceIDs[0], NULL);
  // (6)Obtain device info like obtaining platform info.
  std::vector<std::vector<char>> deviceNames(numDevices);
  for (uint32_t i = 0; i < numDevices; i++) {
    err = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_NAME, 0, NULL, &stringLength);
    deviceNames[i].resize(stringLength);
    err = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_NAME, stringLength,
                          &deviceNames[i][0], NULL);
  }
  // (7)Select device like selecting platform.
  std::vector<std::string> candidateDevices = {"Intel(R) Core(TM)",
                                               "Intel(R) HD", "NVIDIA"};
  uint32_t selectDevice = 0;
  cl_device_id selectedDeviceID = NULL;
  for (uint32_t deviceIndex = 0; deviceIndex < numDevices; deviceIndex++) {
    char *curDeviceName = &deviceNames[deviceIndex][0];
    for (uint32_t charIndex = 0; charIndex < deviceNames[deviceIndex].size();
         charIndex++) {
      uint32_t stringIndex = 0;
      while (curDeviceName[charIndex] ==
             candidateDevices[selectDevice][stringIndex]) {
        charIndex++;
        stringIndex++;
        if (!(charIndex ^ candidateDevices[selectDevice].size())) {
          selectedDeviceID = deviceIDs[deviceIndex];
          break;
        }
      }
      if (selectedDeviceID != NULL)
        break;
    }
    if (selectedDeviceID != NULL)
      break;
  }
  if (selectedDeviceID == NULL)
    std::cout << "Your system has no such OpenCL device "
              << candidateDevices[selectDevice] << "." << std::endl;

  // cl_platform_id first_platform_id;
  // clGetPlatformIDs(1, &first_platform_id, &num_platforms);

  // [> 2. create context <]
  // cl_int err_num;
  // cl_context context = nullptr;
  // cl_context_properties context_prop[] = {
  //     CL_CONTEXT_PLATFORM, (cl_context_properties)first_platform_id, 0};
  // context = clCreateContextFromType(context_prop, CL_DEVICE_TYPE_CPU,
  // nullptr,
  //                                   nullptr, &err_num);
  //
  // [> 3. create command queue <]
  // cl_command_queue command_queue;
  // cl_device_id *devices;
  // size_t device_buffer_size = -1;
  //
  // clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr,
  //                  &device_buffer_size);
  // devices = new cl_device_id[device_buffer_size / sizeof(cl_device_id)];
  // clGetContextInfo(context, CL_CONTEXT_DEVICES, device_buffer_size, devices,
  //                  nullptr);
  // command_queue = clCreateCommandQueue(context, devices[0], 0, nullptr);
  // delete[] devices;
  //
  // [> 4. create program <]
  // std::ifstream kernel_file("vector_add.cl", std::ios::in);
  // std::ostringstream oss;
  //
  // oss << kernel_file.rdbuf();
  // std::string srcStdStr = oss.str();
  // const char *srcStr = srcStdStr.c_str();
  // cl_program program;
  // program = clCreateProgramWithSource(context, 1, (const char **)&srcStr,
  //                                     nullptr, nullptr);
  //
  // [> 5. build program <]
  // clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
  //
  // [> 6. create kernel <]
  // cl_kernel kernel;
  // kernel = clCreateKernel(program, "vector_add", nullptr);
  //
  // [> 7. set input data && create memory object <]
  // float output[DATA_SIZE];
  // float input_x[DATA_SIZE];
  // float input_y[DATA_SIZE];
  // for (int i = 0; i < DATA_SIZE; i++) {
  //   input_x[i] = (float)i;
  //   input_y[i] = (float)(2 * i);
  // }
  //
  // cl_mem mem_object_x;
  // cl_mem mem_object_y;
  // cl_mem mem_object_output;
  // mem_object_x =
  //     clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
  //                    sizeof(float) * DATA_SIZE, input_x, nullptr);
  // mem_object_y =
  //     clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
  //                    sizeof(float) * DATA_SIZE, input_y, nullptr);
  // mem_object_output = clCreateBuffer(
  //     context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE, nullptr,
  //     nullptr);
  //
  // [> 8. set kernel argument <]
  // clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_object_x);
  // clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_object_y);
  // clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_object_output);
  //
  // [> 9. send kernel to execute <]
  // size_t globalWorkSize[1] = {DATA_SIZE};
  // size_t localWorkSize[1] = {1};
  // clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, globalWorkSize,
  //                        localWorkSize, 0, nullptr, nullptr);
  //
  // [> 10. read data from output <]
  // clEnqueueReadBuffer(command_queue, mem_object_output, CL_TRUE, 0,
  //                     DATA_SIZE * sizeof(float), output, 0, nullptr,
  //                     nullptr);
  // for (int i = 0; i < DATA_SIZE; i++) {
  //   std::cout << output[i] << " ";
  // }
  // std::cout << std::endl;
  //
  // [> 11. clean up <]
  // clRetainMemObject(mem_object_x);
  // clRetainMemObject(mem_object_y);
  // clRetainMemObject(mem_object_output);
  // clReleaseCommandQueue(command_queue);
  // clReleaseKernel(kernel);
  // clReleaseProgram(program);
  // clReleaseContext(context);

  return 0;
}
