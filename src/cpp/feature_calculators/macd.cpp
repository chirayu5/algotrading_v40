// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

//*************************************************************
//  MACD (Moving Average Convergence Divergence)
//*************************************************************
            
   else if (var_num == VAR_MACD) {
      short_length = (int) (param1 + 0.5) ;
      long_length = (int) (param2 + 0.5) ;
      n_to_smooth = (int) (param3 + 0.5) ;
      front_bad = long_length + n_to_smooth ;  // Somewhat arbitrary because exponential smoothing
      if (front_bad > n)
         front_bad = n ;
      back_bad = 0 ;

      long_alpha = 2.0 / (long_length + 1.0) ;
      short_alpha = 2.0 / (short_length + 1.0) ;

      long_sum = short_sum = close[0] ;
      output[0] = 0.0 ;   // This would be poorly defined
      for (icase=1 ; icase<n ; icase++) {

         // Compute long-term and short-term exponential smoothing
         long_sum = long_alpha * close[icase] + (1.0 - long_alpha) * long_sum ;
         short_sum = short_alpha * close[icase] + (1.0 - short_alpha) * short_sum ;

         // Compute the normalizing factor, then multiply it by atr to get scaling factor

         diff = 0.5 * (long_length - 1.0) ;     // Center of long block
         diff -= 0.5 * (short_length - 1.0) ;   // Minus center of short block for random walk variance
         denom = sqrt ( fabs(diff) ) ;          // Absolute value should never be needed if careful caller
         k = long_length + n_to_smooth ;
         if (k > icase)                         // ATR requires case at least equal to length
            k = icase ;                         // Which will not be true at the beginning
         denom *= atr ( 0 , icase , k , open , high , low , close ) ;

         // These are the two scalings.  To skip scaling, just use short_sum - long_sum.
         output[icase] = (short_sum - long_sum) / (denom + 1.e-15) ;
         output[icase] = 100.0 * normal_cdf ( 1.0 * output[icase] ) - 50.0 ;
         } // For all cases

      // Smooth and compute differences if requested
      if (n_to_smooth > 1) {
         alpha = 2.0 / (n_to_smooth + 1.0) ;
         smoothed = output[0] ;
         for (icase=1 ; icase<n ; icase++) {
            smoothed = alpha * output[icase] + (1.0 - alpha) * smoothed ;
            output[icase] -= smoothed ;
            } // For all cases
         } // If n_to_smooth > 1
      } // VAR_MACD


//*************************************************************

*/