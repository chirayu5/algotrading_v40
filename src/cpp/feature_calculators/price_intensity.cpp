// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

//*************************************************************
//  PRICE INTENSITY
//*************************************************************

   else if (var_num == VAR_PRICE_INTENSITY) {
      n_to_smooth = (int) (param1 + 0.5) ;
      if (n_to_smooth < 1)
         n_to_smooth = 1 ;
      front_bad = 0 ;          // Number of undefined values at start
      back_bad = 0 ;           // Number of undefined values at end

      // The first bar has no prior bar
      denom = high[0] - low[0] ;
      if (denom < 1.e-60)
         denom = 1.e-60 ;
      output[0] = (close[0] - open[0]) / denom ;

      // Compute raw values
      for (icase=1 ; icase<n ; icase++) {
         denom = high[icase] - low[icase] ;
         if (high[icase] - close[icase-1] > denom)
            denom = high[icase] - close[icase-1] ;
         if (close[icase-1] - low[icase] > denom)
            denom = close[icase-1] - low[icase] ;
         if (denom < 1.e-60)
            denom = 1.e-60 ;
         output[icase] = (close[icase] - open[icase]) / denom ;
         } // For all cases being computed

      // Smooth if requested
      if (n_to_smooth > 1) {
         alpha = 2.0 / (n_to_smooth + 1.0) ;
         smoothed = output[0] ;
         for (icase=1 ; icase<n ; icase++) {
            smoothed = alpha * output[icase] + (1.0 - alpha) * smoothed ;
            output[icase] = smoothed ;
            }
         } // If n_to_smooth

      // Final transformation and mild compression
      for (icase=0 ; icase<n ; icase++)
         output[icase] = 100.0 * normal_cdf ( 0.8 * sqrt((double) n_to_smooth) * output[icase] ) - 50.0 ;

      } // VAR_PRICE_INTENSITY


//*************************************************************

*/