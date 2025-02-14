#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

// Return the first device found in this OpenCL platform.
cl::Device getDefaultDevice();
// Inicialize device and compile kernel code.
void initializeDevice();
// Sequentially performs the operation c[m,n] = a[m,k] * b[k,n].
void seqMultiplyMatrices( const int* a,
                          const int* b,
                          int* c,
                          const size_t m,
                          const size_t n,
                          const size_t k );
// Parallelly performs the operation c[M,N] = a[M,K] * b[K,N].
void parMultiplyMatrices( int* a,
                          int* b,
                          int* c,
                          const size_t m,
                          const size_t n,
                          const size_t k );
// Check if the matrices c1 and c2 are equal.
bool checkEquality( const int* c1,
                    const int* c2,
                    const size_t m,
                    const size_t n );

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
   const int executions = 40;

   /**
    * Prepare input constants related to the dimensions of the matrices.
    * */

   const int m = 1 << 4;
   const int k = 1 << 12;
   const int n = 1 << 4;

   /**
    * Prepare input matrices A and B.
    * */

   const size_t rows_a = m;
   const size_t cols_a = k;
   std::vector< int > a( rows_a * cols_a );
   for( size_t i = 0; i < rows_a * cols_a; i++ ) {
      a[i] = static_cast< int >( i );
   }

   const size_t rows_b = k;
   const size_t cols_b = n;
   std::vector< int > b( rows_b * cols_b );
   for( size_t i = 0; i < rows_b * cols_b; i++ ) {
      b[i] = static_cast< int >( 2 * i );
   }

   /**
    * Prepare sequential and parallel output matrices.
    * */

   const size_t rows_c = m;
   const size_t cols_c = n;
   std::vector< int > cs( rows_c * cols_c );
   std::vector< int > cp( rows_c * cols_c );

   /**
    * Sequentially multiply matrices.
    * */

   start = clock();
   for( int i = 0; i < executions; i++ ) {
      seqMultiplyMatrices( a.data(), b.data(), cs.data(), m, n, k );
   }
   end = clock();
   double seq_time = ( 10e3 * static_cast< double >( end - start ) )
                   / CLOCKS_PER_SEC / executions;

   /**
    * Initialize OpenCL device.
    * */

   initializeDevice();

   /**
    * Parallelly multiply matrices.
    * */

   start = clock();
   for( int i = 0; i < executions; i++ ) {
      parMultiplyMatrices( a.data(), b.data(), cp.data(), m, n, k );
   }
   end = clock();
   double par_time = ( 10e3 * static_cast< double >( end - start ) )
                   / CLOCKS_PER_SEC / executions;

   /**
    * Check if outputs are equal.
    * */

   bool equal = checkEquality( cs.data(), cp.data(), rows_c, cols_c );

   /**
    * Print results.
    * */

   std::cout << "Status: " << ( equal ? "SUCCESS!" : "FAILED!" ) << std::endl;
   std::cout << "Results: \n\tA[0] = " << a[0] << "\n\tB[0] = " << b[0]
             << "\n\tC[0] = " << cp[0] << std::endl;
   std::cout << "Mean execution time: \n\tSequential: " << seq_time
             << " ms;\n\tParallel: " << par_time << " ms." << std::endl;
   std::cout << "Performance gain: "
             << ( 100 * ( seq_time - par_time ) / par_time ) << "\n";
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

   std::ifstream kernel_file( "matrix_multiplication.cl" );
   std::string src( std::istreambuf_iterator< char >( kernel_file ),
                    ( std::istreambuf_iterator< char >() ) );

   /**
    * Compile kernel program which will run on the device.
    * */

   cl::Program::Sources sources{ src.c_str() };
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
 * Sequentially performs the operation c[M,N] = a[M,K] * b[K,N].
 * */

void seqMultiplyMatrices( const int* a,
                          const int* b,
                          int* c,
                          const size_t m,
                          const size_t n,
                          const size_t k ) {
   for( size_t i = 0; i < m; i++ ) {
      for( size_t j = 0; j < n; j++ ) {
         int sum = 0;
         for( size_t z = 0; z < k; z++ ) {
            sum += a[i * k + z] * b[j + z * n];
         }
         c[i * n + j] = sum;
      }
   }
}

/**
 * Parallelly performs the operation c[M,N] = a[M,K] * b[K,N].
 * */

void parMultiplyMatrices( int* a,
                          int* b,
                          int* c,
                          const size_t m,
                          const size_t n,
                          const size_t k ) {

   /**
    * Create buffers and allocate memory on the device.
    * */

   cl::Buffer a_buf(
      context,
      CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
      m * k * sizeof( int ),
      a );
   cl::Buffer b_buf(
      context,
      CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
      k * n * sizeof( int ),
      b );
   cl::Buffer c_buf( context,
                     CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                     m * n * sizeof( int ) );

   /**
    * Set kernel arguments.
    * */

   cl::Kernel kernel( program, "multiplyMatrices" );
   kernel.setArg( 0, a_buf );
   kernel.setArg( 1, b_buf );
   kernel.setArg( 2, c_buf );
   kernel.setArg( 3, sizeof( unsigned int ), &m );
   kernel.setArg( 4, sizeof( unsigned int ), &n );
   kernel.setArg( 5, sizeof( unsigned int ), &k );

   /**
    * Execute the kernel function and collect its result.
    * */

   cl::CommandQueue queue( context, device, CL_QUEUE_PROFILING_ENABLE );
   queue.enqueueNDRangeKernel( kernel, cl::NullRange, cl::NDRange( n, m ) );
   queue.enqueueReadBuffer( c_buf, CL_TRUE, 0, m * n * sizeof( int ), c );
   queue.finish();
}

/**
 * Check if the matrices C1 and C2 are equal.
 * */

bool checkEquality( const int* c1,
                    const int* c2,
                    const size_t m,
                    const size_t n ) {
   for( size_t i = 0; i < m * n; i++ ) {
      if( c1[i] != c2[i] ) {
         return false;
      }
   }
   return true;
}

