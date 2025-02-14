#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <iostream>

int main() {

   /**
    * Search for all the OpenCL platforms available and check
    * if there are any.
    * */

   std::vector< cl::Platform > platforms;
   cl::Platform::get( &platforms );

   if( platforms.empty() ) {
      std::cerr << "No platforms found!" << std::endl;
      return -1;
   }

   /**
    * Search for all the devices on the first platform and check if
    * there are any available.
    * */

   auto platform = platforms.front();
   std::vector< cl::Device > devices;
   platform.getDevices( CL_DEVICE_TYPE_ALL, &devices );

   if( devices.empty() ) {
      std::cerr << "No devices found!" << std::endl;
      return -1;
   }

   /**
    * Select the first device and print its information.
    * */

   auto device = devices.front();
   auto name = device.getInfo< CL_DEVICE_NAME >();
   auto vendor = device.getInfo< CL_DEVICE_VENDOR >();
   auto version = device.getInfo< CL_DEVICE_VERSION >();
   auto work_items = device.getInfo< CL_DEVICE_MAX_WORK_ITEM_SIZES >();
   auto work_groups = device.getInfo< CL_DEVICE_MAX_WORK_GROUP_SIZE >();
   auto compute_units = device.getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >();
   auto global_memory = device.getInfo< CL_DEVICE_GLOBAL_MEM_SIZE >();
   auto local_memory = device.getInfo< CL_DEVICE_LOCAL_MEM_SIZE >();

   std::cout << "OpenCL Device Info:"
             << "\nName: " << name << "\nVendor: " << vendor
             << "\nVersion: " << version << "\nMax size of work-items: ("
             << work_items[0] << "," << work_items[1] << "," << work_items[2]
             << ")"
             << "\nMax size of work-groups: " << work_groups
             << "\nNumber of compute units: " << compute_units
             << "\nGlobal memory size (bytes): " << global_memory
             << "\nLocal memory size per compute unit (bytes): "
             << local_memory / compute_units << std::endl;

   return 0;
}

