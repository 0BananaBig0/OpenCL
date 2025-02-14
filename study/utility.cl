////////////////////////////////////////////////////////////////////////////////
///                           File Name: utili.cl                            ///
///                          Author: Huaxiao Liang                           ///
///                         Mail: 1184903633@qq.com                          ///
///                         02/14/2025-Fri-10:09:10                          ///
////////////////////////////////////////////////////////////////////////////////

unsigned compute_flattened_global_id() {
   unsigned flatten_id = get_global_id( 0 );
   unsigned multiplicand = get_global_size( 0 );
   unsigned work_dims = get_work_dim();   // number of dimensions in use
   for( unsigned dim = 1; dim < work_dims; dim++ ) {
      // flatten_id += get_global_id( dim ) * multiplicand;
      flatten_id
         = mad24( (unsigned)get_global_id( dim ), multiplicand, flatten_id );
      multiplicand = multiplicand * get_global_size( dim );
   }
   return flatten_id;
}

