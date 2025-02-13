#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <time.h>

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

// Return the first device found in this OpenCL platform.
cl::Device getDefaultDevice();
// Inicialize device and compile kernel code.
void initializeDevice();
// Sequentially performs the N-dimensional operation c = a + b.
void seqSumArrays( const int* a, const int* b, int* c, const size_t n );
// Parallelly performs the N-dimensional operation c = a + b.
void parSumArrays( int* a, int* b, int* c, const size_t n );
// Check if the N-dimensional arrays c1 and c2 are equal.
bool checkEquality( const int* c1, const int* c2, const size_t n );

// =================================================================
// ------------------------ Global Variables ------------------------
// =================================================================

cl::Program program;   // The program that will run on the device.
cl::Context context;   // The context which holds the device.
cl::Device device;     // The device where the kernel will run.

// =================================================================
// ------------------------- Main Function -------------------------
// =================================================================

int main() {

   /**
    * Create auxiliary variables.
    * */

   clock_t start, end;
   constexpr int executions = 10;

   /**
    * Prepare input arrays.
    * */

   size_t arrays_dim = 1 << 30;
   std::vector< int > a( arrays_dim );
   std::vector< int > b( arrays_dim );
   for( size_t i = 0; i < arrays_dim; i++ ) {
      a[i] = 2 * static_cast< int >( i );
      b[i] = 3 * static_cast< int >( i );
   }

   /**
    * Prepare sequential and parallel outputs.
    * */

   std::vector< int > cs( arrays_dim );
   std::vector< int > cp( arrays_dim );

   /**
    * Sequentially sum arrays.
    * */

   start = clock();
   for( int i = 0; i < executions; i++ ) {
      seqSumArrays( a.data(), b.data(), cs.data(), arrays_dim );
   }
   end = clock();
   double seq_time = ( 10e3 * static_cast< double >( end - start ) )
                   / CLOCKS_PER_SEC / executions;

   /**
    * Initialize OpenCL device.
    * */

   initializeDevice();

   /**
    * Parallelly sum arrays.
    * */

   start = clock();
   for( int i = 0; i < executions; i++ ) {
      parSumArrays( a.data(), b.data(), cp.data(), arrays_dim );
   }
   end = clock();
   double par_time = ( 10e3 * static_cast< double >( end - start ) )
                   / CLOCKS_PER_SEC / executions;

   /**
    * Check if outputs are equal.
    * */

   bool equal = checkEquality( cs.data(), cp.data(), arrays_dim );

   /**
    * Print results.
    * */

   std::cout << "Status: " << ( equal ? "SUCCESS!" : "FAILED!" ) << std::endl;
   std::cout << "Results: \n\ta[0] = " << a[0] << "\n\tb[0] = " << b[0]
             << "\n\tc[0] = a[0] + b[0] = " << cp[0] << std::endl;
   std::cout << "Mean execution time: \n\tSequential: " << seq_time
             << " ms;\n\tParallel: " << par_time << " ms." << std::endl;
   std::cout << "Performance gain: "
             << ( 100 * ( seq_time - par_time ) / par_time ) << "%\n";
   return 0;
}

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

/**
 * Return the first device found in this OpenCL platform.
 * */

cl::Device getDefaultDevice() {

   /**
    * Search for all the OpenCL platforms available and check
    * if there are any.
    * */

   std::vector< cl::Platform > platforms;
   cl::Platform::get( &platforms );

   if( platforms.empty() ) {
      std::cerr << "No platforms found!" << std::endl;
      exit( 1 );
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
      exit( 1 );
   }

   /**
    * Return the first device found.
    * */

   return devices.front();
}

/**
 * Inicialize device and compile kernel code.
 * */

void initializeDevice() {

   /**
    * Select the first available device.
    * */

   device = getDefaultDevice();

   /**
    * Read OpenCL kernel file as a string.
    * */

   std::ifstream kernel_file( "array_addition.cl" );
   std::string src( std::istreambuf_iterator< char >( kernel_file ),
                    ( std::istreambuf_iterator< char >() ) );

   /**
    * Compile kernel program which will run on the device.
    * */

   cl::Program::Sources sources{ src };
   context = cl::Context( device );
   program = cl::Program( context, sources );

   auto err = program.build();
   if( err != CL_BUILD_SUCCESS ) {
      std::cerr << "Error!\nBuild Status: "
                << program.getBuildInfo< CL_PROGRAM_BUILD_STATUS >( device )
                << "\nBuild Log:\t "
                << program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( device )
                << std::endl;
      exit( 1 );
   }
}

/**
 * Sequentially performs the N-dimensional operation c = a + b.
 * */

void seqSumArrays( const int* a, const int* b, int* c, const size_t n ) {
   for( size_t i = 0; i < n; i++ ) {
      c[i] = a[i] + b[i];
   }
}

/**
 * Parallelly performs the N-dimensional operation c = a + b.
 * */

void parSumArrays( int* a, int* b, int* c, const size_t n ) {

   /**
    * Create buffers and allocate memory on the device.
    * */

   cl::Buffer a_buf(
      context,
      CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
      n * sizeof( int ),
      a );
   cl::Buffer b_buf(
      context,
      CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
      n * sizeof( int ),
      b );
   cl::Buffer c_buf( context,
                     CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                     n * sizeof( int ) );

   /**
    * Set kernel arguments.
    * */

   cl::Kernel kernel( program, "sumArrays" );
   kernel.setArg( 0, a_buf );
   kernel.setArg( 1, b_buf );
   kernel.setArg( 2, c_buf );

   /**
    * Execute the kernel function and collect its result.
    * */

   cl::CommandQueue queue( context, device );
   queue.enqueueNDRangeKernel( kernel, cl::NullRange, cl::NDRange( n ) );
   queue.enqueueReadBuffer( c_buf, CL_TRUE, 0, n * sizeof( int ), c );
}

/**
 * Check if the N-dimensional arrays c1 and c2 are equal.
 * */

bool checkEquality( const int* c1, const int* c2, const size_t n ) {
   for( size_t i = 0; i < n; i++ ) {
      if( c1[i] != c2[i] ) {
         return false;
      }
   }
   return true;
}

