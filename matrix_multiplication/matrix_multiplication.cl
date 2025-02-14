/**
 * This kernel function multiplies two matrices a[M,K] and b[K,N].
 **/

__kernel void multiplyMatrices( const __global int* a,
                                const __global int* b,
                                __global int* c,
                                const int m,
                                const int n,
                                const int k ) {

   /**
    * Get work-item identifiers.
    **/

   int col_index = (int)get_global_id( 0 );
   int row_index = (int)get_global_id( 1 );
   int index = ( row_index * n ) + col_index;

   /**
    * Compute element c[rowIndex, colIndex].
    **/

   int sum = 0;
   for( int z = 0; z < k; z++ ) {
      sum += a[row_index * k + z] * b[z * n + col_index];
   }
   c[index] = sum;
}

