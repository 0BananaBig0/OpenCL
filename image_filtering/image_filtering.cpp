#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <string.h>
#include <time.h>

#include "CImg.h"
using namespace cimg_library;

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

// Sequentially convert an RGB image to grayscale.
void seqRgb2Gray( unsigned int img_width,
                  unsigned int img_height,
                  const unsigned char* r_channel,
                  const unsigned char* g_channel,
                  const unsigned char* b_channel,
                  unsigned char* gray_img );

// Sequentially convolve an image with a filter.
void seqConvolve( unsigned int img_width,
                  unsigned int img_height,
                  unsigned int mask_size,
                  const unsigned char* input_img,
                  const float* mask,
                  unsigned char* output_img );

// Sequentially filter an image.
void seqFilter( unsigned int img_width,
                unsigned int img_height,
                unsigned int lp_mask_size,
                unsigned int hp_mask_size,
                unsigned char* input_rchannel,
                unsigned char* input_gchannel,
                unsigned char* input_bchannel,
                float* lp_mask,
                float* hp_mask,
                unsigned char* output_img );

// Check if the images img1 and img2 are equal.
bool checkEquality( const unsigned char* img1,
                    const unsigned char* img2,
                    const unsigned int m,
                    const unsigned int n );

// Display unsigned char matrix as an image.
void displayImg( const unsigned char* img,
                 unsigned int img_width,
                 unsigned int img_height );

// =================================================================
// ------------------------ OpenCL Functions -----------------------
// =================================================================

// Return a device found in this OpenCL platform.
cl::Device getDefaultDevice();

// Inicialize device and compile kernel code.
void initializeDevice();

// Parallelly filter an image.
void parFilter( unsigned int img_width,
                unsigned int img_height,
                unsigned int lp_mask_size,
                unsigned int hp_mask_size,
                unsigned char* input_rchannel,
                unsigned char* input_gchannel,
                unsigned char* input_bchannel,
                float* lp_mask,
                float* hp_mask,
                unsigned char* output_img );

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

   /**
    * Load input image.
    * */

   CImg< unsigned char > cimg( "input_img.jpg" );
   unsigned char* input_img = cimg.data();
   unsigned int img_width = static_cast< unsigned int >( cimg.width() );
   unsigned int img_height = static_cast< unsigned int >( cimg.height() );
   unsigned char* input_rchannel = &input_img[0];
   unsigned char* input_gchannel = &input_img[img_width * img_height];
   unsigned char* input_bchannel = &input_img[2 * img_width * img_height];

   /**
    * Create a low-pass filter mask.
    * */

   const int lp_mask_size = 5;
   float lp_mask[lp_mask_size][lp_mask_size] = {
      { .04f, .04f, .04f, .04f, .04f },
      { .04f, .04f, .04f, .04f, .04f },
      { .04f, .04f, .04f, .04f, .04f },
      { .04f, .04f, .04f, .04f, .04f },
      { .04f, .04f, .04f, .04f, .04f },
   };
   float* lp_mask_data = &lp_mask[0][0];

   /**
    * Create a high-pass filter mask.
    * */

   const int hp_mask_size = 5;
   float hp_mask[hp_mask_size][hp_mask_size] = {
      { -1, -1, -1, -1, -1 },
      { -1, -1, -1, -1, -1 },
      { -1, -1, 24, -1, -1 },
      { -1, -1, -1, -1, -1 },
      { -1, -1, -1, -1, -1 },
   };
   float* hp_mask_data = &hp_mask[0][0];

   /**
    * Allocate memory for the output images.
    * */

   unsigned char* seq_filtered_img = static_cast< unsigned char* >(
      malloc( img_width * img_height * sizeof( unsigned char ) ) );
   unsigned char* par_filtered_img = static_cast< unsigned char* >(
      malloc( img_width * img_height * sizeof( unsigned char ) ) );

   /**
    * Sequentially convolve filter over image.
    * */

   start = clock();
   seqFilter( img_width,
              img_height,
              lp_mask_size,
              hp_mask_size,
              input_rchannel,
              input_gchannel,
              input_bchannel,
              lp_mask_data,
              hp_mask_data,
              seq_filtered_img );
   end = clock();
   double seq_time
      = ( 10e3 * static_cast< double >( end - start ) ) / CLOCKS_PER_SEC;

   /**
    * Initialize OpenCL device.
    */

   initializeDevice();

   /**
    * Parallelly convolve filter over image.
    * */

   start = clock();
   parFilter( img_width,
              img_height,
              lp_mask_size,
              hp_mask_size,
              input_rchannel,
              input_gchannel,
              input_bchannel,
              lp_mask_data,
              hp_mask_data,
              par_filtered_img );
   end = clock();
   double par_time
      = ( 10e3 * static_cast< double >( end - start ) ) / CLOCKS_PER_SEC;

   /**
    * Check if outputs are equal.
    * */

   bool equal = checkEquality( seq_filtered_img,
                               par_filtered_img,
                               img_width,
                               img_height );

   /**
    * Print results.
    */

   std::cout << "Status: " << ( equal ? "SUCCESS!" : "FAILED!" ) << std::endl;
   std::cout << "Mean execution time: \n\tSequential: " << seq_time
             << " ms;\n\tParallel: " << par_time << " ms." << std::endl;
   std::cout << "Performance gain: "
             << ( 100 * ( seq_time - par_time ) / par_time ) << "\n";

   /**
    * Display filtered image.
    * */

   displayImg( par_filtered_img, img_width, img_height );
   return 0;
}

// =================================================================
// ------------------------ OpenCL Functions -----------------------
// =================================================================

/**
 * Return a device found in this OpenCL platform.
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
    * Search for all the devices on the first platform
    * and check if there are any available.
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

   std::ifstream kernel_file( "image_filtering.cl" );
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
 * Parallelly filter an image.
 */

void parFilter( unsigned int img_width,
                unsigned int img_height,
                unsigned int lp_mask_size,
                unsigned int hp_mask_size,
                unsigned char* input_rchannel,
                unsigned char* input_gchannel,
                unsigned char* input_bchannel,
                float* lp_mask,
                float* hp_mask,
                unsigned char* output_img ) {

   /**
    * Create buffers and allocate memory on the device.
    * */

   cl::Buffer input_rchannel_buf(
      context,
      CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
      img_width * img_height * sizeof( unsigned char ),
      input_rchannel );
   cl::Buffer input_gchannel_buf(
      context,
      CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
      img_width * img_height * sizeof( unsigned char ),
      input_gchannel );
   cl::Buffer input_bchannel_buf(
      context,
      CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
      img_width * img_height * sizeof( unsigned char ),
      input_bchannel );
   cl::Buffer gray_output_buf(
      context,
      CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
      img_width * img_height * sizeof( unsigned char ) );
   cl::Buffer lp_mask_buf(
      context,
      CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
      lp_mask_size * lp_mask_size * sizeof( float ),
      lp_mask );
   cl::Buffer hp_mask_buf(
      context,
      CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
      hp_mask_size * hp_mask_size * sizeof( float ),
      hp_mask );
   cl::Buffer lp_output_buf( context,
                             CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                             img_width * img_height * sizeof( unsigned char ) );
   cl::Buffer hp_output_buf( context,
                             CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                             img_width * img_height * sizeof( unsigned char ) );

   /**
    * Initialize grayscale kernel.
    * */

   cl::Kernel gray_kernel( program, "rgb2gray" );
   gray_kernel.setArg( 0, input_rchannel_buf );
   gray_kernel.setArg( 1, input_gchannel_buf );
   gray_kernel.setArg( 2, input_bchannel_buf );
   gray_kernel.setArg( 3, gray_output_buf );

   /**
    * Initialize low-pass filter kernel.
    * */
   cl::Kernel lp_kernel( program, "filterImageWithCache" );
   lp_kernel.setArg( 0, sizeof( unsigned int ), &lp_mask_size );
   lp_kernel.setArg( 1, gray_output_buf );
   lp_kernel.setArg( 2, lp_mask_buf );
   lp_kernel.setArg( 3, lp_output_buf );

   /**
    * Initialize high-pass filter kernel.
    * */

   cl::Kernel hp_kernel( program, "filterImageWithCache" );
   hp_kernel.setArg( 0, sizeof( unsigned int ), &hp_mask_size );
   hp_kernel.setArg( 1, lp_output_buf );
   hp_kernel.setArg( 2, hp_mask_buf );
   hp_kernel.setArg( 3, hp_output_buf );

   /**
    * Execute kernel functions and collect the final result.
    * */

   cl::CommandQueue queue( context, device );
   queue.enqueueNDRangeKernel( gray_kernel,
                               cl::NullRange,
                               cl::NDRange( img_width, img_height ) );
   queue.enqueueNDRangeKernel( lp_kernel,
                               cl::NullRange,
                               cl::NDRange( img_width, img_height ),
                               cl::NDRange( 16, 16 ) );
   queue.enqueueNDRangeKernel( hp_kernel,
                               cl::NullRange,
                               cl::NDRange( img_width, img_height ),
                               cl::NDRange( 16, 16 ) );
   queue.enqueueReadBuffer( hp_output_buf,
                            CL_TRUE,
                            0,
                            img_width * img_height * sizeof( unsigned char ),
                            output_img );
}

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

/**
 * Sequentially convert an RGB image to grayscale.
 */

void seqRgb2Gray( unsigned int img_width,
                  unsigned int img_height,
                  const unsigned char* r_channel,
                  const unsigned char* g_channel,
                  const unsigned char* b_channel,
                  unsigned char* gray_img ) {

   /**
    * Declare the current index variable.
    */

   unsigned int idx;

   /**
    * Loop over input image pixels.
    */

   for( unsigned int i = 0; i < img_width; i++ ) {
      for( unsigned int j = 0; j < img_height; j++ ) {

         /**
          * Compute average pixel.
          */

         idx = i + j * img_width;
         gray_img[idx]
            = ( r_channel[idx] + g_channel[idx] + b_channel[idx] ) / 3;
      }
   }
}

/**
 * Sequentially convolve an image with a filter mask.
 */

void seqConvolve( unsigned int img_width,
                  unsigned int img_height,
                  unsigned int mask_size,
                  const unsigned char* input_img,
                  const float* mask,
                  unsigned char* output_img ) {
   /**
    * Loop through input image.
    * */

   for( size_t i = 0; i < img_width; i++ ) {
      for( size_t j = 0; j < img_height; j++ ) {

         /**
          * Check if the mask cannot be applied to the
          * current image pixel.
          * */

         if( i < mask_size / 2 || j < mask_size / 2
             || i >= img_width - mask_size / 2
             || j >= img_height - mask_size / 2 ) {
            output_img[i + j * img_width] = 0;
            continue;
         }

         /**
          * Apply mask based on the neighborhood of pixel inputImg(j,i).
          * */

         int out_sum = 0;
         for( size_t k = 0; k < mask_size; k++ ) {
            for( size_t l = 0; l < mask_size; l++ ) {
               size_t col_idx = i - mask_size / 2 + k;
               size_t row_idx = j - mask_size / 2 + l;
               size_t mask_idx
                  = ( mask_size - 1 - k ) + ( mask_size - 1 - l ) * mask_size;
               out_sum += static_cast< int >(
                  static_cast< float >(
                     input_img[row_idx * img_width + col_idx] )
                  * mask[mask_idx] );
            }
         }

         /**
          * Update output pixel.
          * */

         if( out_sum < 0 ) {
            output_img[i + j * img_width] = 0;
         } else if( out_sum > 255 ) {
            output_img[i + j * img_width] = 255;
         } else {
            output_img[i + j * img_width]
               = static_cast< unsigned char >( out_sum );
         }
      }
   }
}

/**
 * Sequentially filter an image.
 */

void seqFilter( unsigned int img_width,
                unsigned int img_height,
                unsigned int lp_mask_size,
                unsigned int hp_mask_size,
                unsigned char* input_rchannel,
                unsigned char* input_gchannel,
                unsigned char* input_bchannel,
                float* lp_mask,
                float* hp_mask,
                unsigned char* output_img ) {

   /**
    * Convert input image to grayscale.
    */

   unsigned char* gray_out = static_cast< unsigned char* >(
      malloc( img_width * img_height * sizeof( unsigned char ) ) );
   seqRgb2Gray( img_width,
                img_height,
                input_rchannel,
                input_gchannel,
                input_bchannel,
                gray_out );

   /**
    * Apply the low-pass filter.
    */

   unsigned char* lp_out = static_cast< unsigned char* >(
      malloc( img_width * img_height * sizeof( unsigned char ) ) );
   seqConvolve( img_width,
                img_height,
                lp_mask_size,
                gray_out,
                lp_mask,
                lp_out );

   /**
    * Apply the high-pass filter.
    */

   seqConvolve( img_width,
                img_height,
                hp_mask_size,
                lp_out,
                hp_mask,
                output_img );
}

/**
 * Display unsigned char matrix as an image.
 * */

void displayImg( const unsigned char* img,
                 unsigned int img_width,
                 unsigned int img_height ) {

   /**
    * Create C_IMG object.
    * */

   CImg< unsigned char > cimg( img_width, img_height );

   /**
    * Transfer image data to C_IMG object.
    * */

   for( unsigned int i = 0; i < img_width; i++ ) {
      for( unsigned int j = 0; j < img_height; j++ ) {
         cimg( i, j ) = img[i + img_width * j];
      }
   }

   /**
    * Display image.
    * */

   cimg.display();
}

/**
 * Check if the images img1 and img2 are equal.
 * */

bool checkEquality( const unsigned char* img1,
                    const unsigned char* img2,
                    const unsigned int m,
                    const unsigned int n ) {
   for( unsigned int i = 0; i < m * n; i++ ) {
      if( img1[i] != img2[i] ) {
         return false;
      }
   }
   return true;
}

