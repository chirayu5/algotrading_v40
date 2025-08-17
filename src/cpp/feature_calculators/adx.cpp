// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

//*************************************************************
//  ADX
//*************************************************************

   else if (var_num == VAR_ADX) {
      lookback = (int) (param1 + 0.5) ;
      front_bad = 2 * lookback - 1 ; // Number of undefined values at start
      back_bad = 0 ;                 // Number of undefined values at end

      output[0] = 0 ;  // This totally invalid case gets a neutral value

      // Primary initialization sums DMplus, DMminus, and ATR over first lookback prices
      // This gives the first (not yet smoothed or fully valid) ADX

      DMSplus = DMSminus = ATR = 0.0 ;
      for (icase=1 ; icase<=lookback ; icase++) {
         DMplus = high[icase] - high[icase-1] ;  // Upward move
         DMminus = low[icase-1] - low[icase] ;   // Downward move
         if (DMplus >= DMminus)          // Pick whichever is larger
            DMminus = 0.0 ;              // and discard the smaller
         else
            DMplus = 0.0 ;
         if (DMplus < 0.0)               // Moves cannot be negative
            DMplus = 0.0 ;
         if (DMminus < 0.0)
            DMminus = 0.0 ;
         DMSplus += DMplus ;
         DMSminus += DMminus ;
         term = high[icase] - low[icase] ;  // Now cumulate ATR
         if (high[icase] - close[icase-1] > term)
            term = high[icase] - close[icase-1] ;
         if (close[icase-1] - low[icase] > term)
            term = close[icase-1] - low[icase] ;
         ATR += term ;
         // Officially we don't need these computations yet, but might as well put them in output during warmup
         DIplus = DMSplus / (ATR + 1.e-10) ;
         DIminus = DMSminus / (ATR + 1.e-10) ;
         ADX = fabs ( DIplus - DIminus ) / (DIplus + DIminus + 1.e-10) ; // When loop is done this will be first ADX
         output[icase] = 100.0 * ADX ;  // Not very settled down yet, but fairly valid
#if 0
         char msg[256] ;
         sprintf_s ( msg, "%5d DM+=%6.2lf DM-=%6.2lf DMS+=%6.2lf DMS-=%6.2lf term=%6.2lf ATR=%7.2lf DI+=%8.3lf DI-=%8.3lf ADX=%.4lf",
                     icase, DMplus, DMminus, DMSplus, DMSminus, term, ATR, DIplus, DIminus, ADX ) ;
         MEMTEXT ( msg ) ;
#endif
         }

      // Secondary initialization gets the next lookback-1 values,
      // adding them into ADX so we can get a simple average.
      // But from here on we use exponential smoothing for DMSplus, DMSminus, and ATR

      for (icase=lookback+1 ; icase<2*lookback ; icase++) {
         DMplus = high[icase] - high[icase-1] ;  // Upward move
         DMminus = low[icase-1] - low[icase] ;   // Downward move
         if (DMplus >= DMminus)          // Pick whichever is larger
            DMminus = 0.0 ;              // and discard the smaller
         else
            DMplus = 0.0 ;
         if (DMplus < 0.0)               // Moves cannot be negative
            DMplus = 0.0 ;
         if (DMminus < 0.0)
            DMminus = 0.0 ;
         DMSplus = (lookback - 1.0) / lookback * DMSplus + DMplus ;
         DMSminus = (lookback - 1.0) / lookback * DMSminus + DMminus ;
         term = high[icase] - low[icase] ;  // Now cumulate ATR
         if (high[icase] - close[icase-1] > term)
            term = high[icase] - close[icase-1] ;
         if (close[icase-1] - low[icase] > term)
            term = close[icase-1] - low[icase] ;
         ATR = (lookback - 1.0) / lookback * ATR + term ;
         DIplus = DMSplus / (ATR + 1.e-10) ;
         DIminus = DMSminus / (ATR + 1.e-10) ;
         ADX += fabs ( DIplus - DIminus ) / (DIplus + DIminus + 1.e-10) ;    // Cumulate for simple average
         output[icase] = 100.0 * ADX / (icase-lookback+1) ;  // Not very settled down yet, but fairly valid
#if 0
         sprintf_s ( msg, "%5d DM+=%6.2lf DM-=%6.2lf DMS+=%6.2lf DMS-=%6.2lf term=%6.2lf ATR=%7.2lf DI+=%8.3lf DI-=%8.3lf ADX=%.4lf",
                     icase, DMplus, DMminus, DMSplus, DMSminus, term, ATR, DIplus, DIminus, output[icase] ) ;
         MEMTEXT ( msg ) ;
#endif
         }
      ADX /= lookback ;  // First valid value; we put it in output above

      // Whew, that was a chore!  And sadly, just using exponential smoothing
      // all along would have not only been a LOT simpler but probably even better
      // due to the faster response of exponential smoothing.
      // But I felt it important to follow Wilder's original algorithm.
      // In any case, it's a moot point because this was just initialization.
      // Either method would give similar results soon after this initialization.
      // From here on we use exponential smoothing for everything.

      for (icase=2*lookback ; icase<n ; icase++) {
         DMplus = high[icase] - high[icase-1] ;  // Upward move
         DMminus = low[icase-1] - low[icase] ;   // Downward move
         if (DMplus >= DMminus)          // Pick whichever is larger
            DMminus = 0.0 ;              // and discard the smaller
         else
            DMplus = 0.0 ;
         if (DMplus < 0.0)               // Moves cannot be negative
            DMplus = 0.0 ;
         if (DMminus < 0.0)
            DMminus = 0.0 ;
         DMSplus = (lookback - 1.0) / lookback * DMSplus + DMplus ;
         DMSminus = (lookback - 1.0) / lookback * DMSminus + DMminus ;
         term = high[icase] - low[icase] ;  // Now cumulate ATR
         if (high[icase] - close[icase-1] > term)
            term = high[icase] - close[icase-1] ;
         if (close[icase-1] - low[icase] > term)
            term = close[icase-1] - low[icase] ;
         ATR = (lookback - 1.0) / lookback * ATR + term ;
         DIplus = DMSplus / (ATR + 1.e-10) ;
         DIminus = DMSminus / (ATR + 1.e-10) ;
#if 0
         sprintf_s ( msg, "%5d DM+=%6.2lf DM-=%6.2lf DMS+=%6.2lf DMS-=%6.2lf term=%6.2lf ATR=%7.2lf",
                     icase, DMplus, DMminus, DMSplus, DMSminus, term, ATR ) ;
         MEMTEXT ( msg ) ;
#endif
         term = fabs ( DIplus - DIminus ) / (DIplus + DIminus + 1.e-10) ; // This ADX
         ADX = (lookback - 1.0) / lookback * ADX + term / lookback ;
#if 0
         sprintf_s ( msg, "   DI+=%8.3lf DI-=%8.3lf ADX=%.4lf", DIplus, DIminus, ADX ) ;
         MEMTEXT ( msg ) ;
#endif
         output[icase] = 100.0 * ADX ;
         }  // For all remaining cases, which are valid
      } // VAR_ADX


//*************************************************************

*/