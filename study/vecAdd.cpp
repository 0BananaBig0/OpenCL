/*************************************************************************
  > File Name: vecAdd.cpp
  > Author: 16hxliang3
  > Mail: 16hxliang3@stu.edu.cn
  > Created Time: Mon 08 Aug 2022 09:16:47 PM CST
 ************************************************************************/

#include <fstream>
#include <iostream>
#include <sstream>

#include <CL/cl.h>

const int DATA_SIZE = 9999;

int main(void) {
   /* 1. get platform & device information */
   cl_uint num_platforms;
   cl_platform_id first_platform_id;
   clGetPlatformIDs(1, &first_platform_id, &num_platforms);

   /* 2. create context */
   cl_int err_num;
   cl_context context = nullptr;
   cl_context_properties context_prop[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)first_platform_id, 0};
   context = clCreateContextFromType(context_prop, CL_DEVICE_TYPE_CPU, nullptr,
                                     nullptr, &err_num);

   /* 3. create command queue */
   cl_command_queue command_queue;
   cl_device_id *devices;
   size_t device_buffer_size = -1;

   clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr,
                    &device_buffer_size);
   devices = new cl_device_id[device_buffer_size / sizeof(cl_device_id)];
   clGetContextInfo(context, CL_CONTEXT_DEVICES, device_buffer_size, devices,
                    nullptr);
   command_queue =
      clCreateCommandQueueWithProperties(context, devices[0], nullptr, nullptr);
   delete[] devices;

   /* 4. create program */
   std::ifstream kernel_file("vecAdd.cl", std::ios::in);
   std::ostringstream oss;

   oss << kernel_file.rdbuf();
   std::string srcStdStr = oss.str();
   const char *srcStr = srcStdStr.c_str();
   cl_program program;
   program = clCreateProgramWithSource(context, 1, (const char **)&srcStr,
                                       nullptr, nullptr);

   /* 5. build program */
   clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);

   /* 6. create kernel */
   cl_kernel kernel;
   kernel = clCreateKernel(program, "vector_add", nullptr);

   /* 7. set input data && create memory object */
   float output[DATA_SIZE];
   float input_x[DATA_SIZE];
   float input_y[DATA_SIZE];
   for(int i = 0; i < DATA_SIZE; i++) {
      input_x[i] = (float)i;
      input_y[i] = (float)(2 * i);
   }

   cl_mem mem_object_x;
   cl_mem mem_object_y;
   cl_mem mem_object_output;
   mem_object_x =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * DATA_SIZE, input_x, nullptr);
   mem_object_y =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * DATA_SIZE, input_y, nullptr);
   mem_object_output = clCreateBuffer(
      context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE, nullptr, nullptr);

   /* 8. set kernel argument */
   clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_object_x);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_object_y);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_object_output);

   /* 9. send kernel to execute */
   size_t globalWorkSize[1] = {DATA_SIZE};
   size_t localWorkSize[1] = {1};
   clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, globalWorkSize,
                          localWorkSize, 0, nullptr, nullptr);

   /* 10. read data from output */
   clEnqueueReadBuffer(command_queue, mem_object_output, CL_TRUE, 0,
                       DATA_SIZE * sizeof(float), output, 0, nullptr, nullptr);
   for(int i = 0; i < DATA_SIZE; i++) {
      std::cout << output[i] << " ";
   }
   std::cout << std::endl;

   /* 11. clean up */
   clRetainMemObject(mem_object_x);
   clRetainMemObject(mem_object_y);
   clRetainMemObject(mem_object_output);
   clReleaseCommandQueue(command_queue);
   clReleaseKernel(kernel);
   clReleaseProgram(program);
   clReleaseContext(context);

   return 0;
}
