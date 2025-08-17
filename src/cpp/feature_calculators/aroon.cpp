// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

//*************************************************************
//  AROON_?
//*************************************************************

   else if (var_num == VAR_AROON_UP  ||  var_num == VAR_AROON_DOWN  ||  var_num == VAR_AROON_DIFF) {
      lookback = (int) (param1 + 0.5) ;
      front_bad = lookback ;   // Number of undefined values at start
      back_bad = 0 ;           // Number of undefined values at end

      // Even though front_bad is set to lookback,
      // we can find reasonable values for all but the first bar.
      if (var_num == VAR_AROON_UP  ||  var_num == VAR_AROON_DOWN)
         output[0] = 50.0 ;    // Set only undefined value to neutral
      else
         output[0] = 0.0 ;     // Set only undefined value to neutral

      for (icase=1 ; icase<n ; icase++) {

         if (var_num == VAR_AROON_UP  ||  var_num == VAR_AROON_DIFF) {
            imax = icase ;                              // Keeps track of bar with high
            xmax = high[icase] ;
            for (i=icase-1 ; i>=icase-lookback ; i--) { // We actually examine lookback+1 prices
               if (i < 0)                               // Current case not included in lookback
                  break ;

               if (high[i] > xmax) {
                  xmax = high[i] ;
                  imax = i ;
                  }
               }
            }

         if (var_num == VAR_AROON_DOWN  ||  var_num == VAR_AROON_DIFF) {
            imin = icase ;
            xmin = low[icase] ;
            for (i=icase-1 ; i>=icase-lookback ; i--) {
               if (i < 0)
                  break ;

               if (low[i] < xmin) {
                  xmin = low[i] ;
                  imin = i ;
                  }
               }
            }

         if (var_num == VAR_AROON_UP)
            output[icase] = 100.0 * (lookback - (icase - imax)) / lookback ;
         else if (var_num == VAR_AROON_DOWN)
            output[icase] = 100.0 * (lookback - (icase - imin)) / lookback ;
         else {
            max_val = 100.0 * (lookback - (icase - imax)) / lookback ;
            min_val = 100.0 * (lookback - (icase - imin)) / lookback ;
            output[icase] = max_val - min_val ;
            }
         } // For icase, computing all values
      } // VAR_AROON_?


//*************************************************************

*/