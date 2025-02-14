/**
 * This kernel function efficiently multiplies two matrices a[m,k] and b[k,n]
 * by caching submatrices from those input matrices in the device local memory.
 */

__kernel void multiplyMatricesWithCache( const __global int* a,
                                         const __global int* b,
                                         __global int* c,
                                         const unsigned int m,
                                         const unsigned int n,
                                         const unsigned int k ) {

   /**
    * Declare the size of each submatrix (it must be
    * the same work-group size declared in the host code).
    */

   const int sub_size = 16;

   /**
    * Get work-item identifiers.
    */

   int col_index = (int)get_local_id( 0 );
   int row_index = (int)get_local_id( 1 );
   int global_col_index = (int)get_global_id( 0 );
   int global_row_index = (int)get_global_id( 1 );
   int index = ( global_row_index * (int)n ) + global_col_index;

   /**
    * Create submatrices that will cache the matrices A and B in local memory.
    */

   __local int a_sub[sub_size][sub_size];
   __local int b_sub[sub_size][sub_size];

   /**
    * Initialize accumulator register.
    */

   int sum = 0;

   /**
    * Loop over all submatrices.
    */

   // Determine repeated times.
   // Utilize sliding window technique to calcuate the target result.
   const int n_sub = (int)k / sub_size;
   for( int i = 0; i < n_sub; i++ ) {

      /**
       * Load submatrices into local memory.
       */

      const int s_col = sub_size * i + col_index;
      const int s_row = sub_size * i + row_index;
      a_sub[row_index][col_index] = a[global_row_index * (int)k + s_col];
      b_sub[row_index][col_index] = b[s_row * (int)n + global_col_index];

      /**
       * Synchronize all work-items in this work-group.
       */

      barrier( CLK_LOCAL_MEM_FENCE );

      /**
       * Perform the computation for a single submatrix.
       */

      for( int j = 0; j < sub_size; j++ ) {
         sum += a_sub[row_index][j] * b_sub[j][col_index];
      }

      /**
       * Synchronize all work-items in this work-group.
       */

      barrier( CLK_LOCAL_MEM_FENCE );
   }

   /**
    * Store the final result in the matrix C.
    */

   c[index] = sum;
}

