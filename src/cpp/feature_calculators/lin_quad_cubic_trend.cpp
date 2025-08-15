// clang-format off
/*
// RAW UNTOUCHED CODE FROM ORIGINAL

//*************************************************************
//  LINEAR/QUADRATIC/CUBIC TREND
//*************************************************************

   else if (var_num == VAR_LINEAR_TREND  ||  var_num == VAR_QUADRATIC_TREND  ||  var_num == VAR_CUBIC_TREND) {
      lookback = (int) (param1 + 0.5) ;    // Lookback for trend
      atr_length = (int) (param2 + 0.5) ;  // Lookback for ATR normalization (should greatly exceed lookback)
      front_bad = ((lookback-1) > atr_length) ? (lookback-1) : atr_length ;
      if (front_bad > n)
         front_bad = n ;
      back_bad = 0 ;

      legendre_3 ( lookback , work1 , work2 , work3 ) ;  // Compute all 3 even though we need just 1.  Fast.

      for (icase=0 ; icase<front_bad ; icase++)
         output[icase] = 0.0 ;   // Set undefined values to neutral value

      for (icase=front_bad ; icase<n ; icase++) {
         if (var_num == VAR_LINEAR_TREND)  // Choose the correct set of Legendre coefficients
            dptr = work1 ;
         else if (var_num == VAR_QUADRATIC_TREND)
            dptr = work2 ;
         else if (var_num == VAR_CUBIC_TREND)
            dptr = work3 ;

         // The regression coefficient (in dot_prod) is the dot product of the
         // log prices with the Legendre polynomial coefficients.

         dot_prod = 0.0 ;
         mean = 0.0 ;   // We need this for rsq
         for (k=icase-lookback+1 ; k<=icase ; k++) {  // The trend lookback window
            price = log ( close[k] ) ;
            mean += price ;
            dot_prod += price * *dptr++ ;             // Cumulate dot product
            }
         mean /= lookback ;
         dptr -= lookback ;   // Reset coefs pointer to start for finding rsq

         // Dot_prod is regression coef (log price change per unit X change)
         // Total X change over window is 1 - (-1) = 2 (domain of Legendre polynomial)
         // So dot_prod * 2 is the fitted change over the window
         // Denom is change over window based on ATR if all changes exactly match Legendre polynomial
         // Thus, the basic indicator (prior to rsq and compression) is the ratio
         // of the achieved fitted change to the theoretical change based on ATR.

         k = lookback - 1 ;
         if (lookback == 2)
            k = 2 ;
         denom = atr ( 1 , icase , atr_length , open , high , low , close ) * k ;
         output[icase] = dot_prod * 2.0 / (denom + 1.e-60) ;  // Change over window / window ATR path

         // At this point, output[icase] is the ratio of the observed change to the
         // theoretical change implied by ATR that follows the tested path.
         // Compute R-square for degrading the indicator if it is a poor fit

         yss = rsq = 0.0 ;
         for (k=icase-lookback+1 ; k<=icase ; k++) {  // The trend lookback window
            price = log ( close[k] ) ;
            diff = price - mean ;       // Y offset from its mean
            yss += diff * diff ;        // Cumulate Y sum of squares
            pred = dot_prod * *dptr++ ; // Regression coefficient times X is predicted Y offset
            diff = diff - pred ;        // Y offset from mean minus predicted offset
            rsq += diff * diff ;        // Sum the squared error 
            }
         rsq = 1.0 - rsq / (yss + 1.e-60) ;     // Definition of R-square
         if (rsq < 0.0)                 // Should never happen
            rsq = 0.0 ;
         output[icase] *= rsq ;         // Degrade the indicator if it is a poor fit

         output[icase] = 100.0 * normal_cdf ( output[icase] ) - 50.0 ;  // Weakly compress outliers
         } // For all cases being computed

      } // VAR_LINEAR/QUADRATIC/CUBIC_TREND


//*************************************************************

*/