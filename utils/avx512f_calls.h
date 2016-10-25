/* File: avx512f_calls.h */
/*
  This file is a part of the Corrfunc package
  Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/manodeep/Corrfunc/
*/

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "function_precision.h" 

#define PREFETCH(mem)        asm ("prefetcht0 %0"::"m"(mem))

#if defined(__GNUC__) || defined(__GNUG__)
#define AVX512F_BIT_COUNT_INT(X)                __builtin_popcount(X)
#else
#define AVX512F_BIT_COUNT_INT(X)                _popcnt32(X)
#endif


#define AVX512F_BIT_COUNT_LONG(X)               _popcnt64(X)
#define AVX512F_BIT_COUNT_UNSIGNED_INT(X)       _mm_popcnt_u32(X)
#define AVX512F_BIT_COUNT_UNSIGNED_LONG(X)      _mm_popcnt_u64(X)
#define AVX512F_SET_INT(X)                      _mm512_set1_epi32(X)

#define AVX512F_BLEND_INTS_WITH_MASK(MASK, FALSEVALUE, TRUEVALUE)  _mm512_mask_blend_epi32(MASK, FALSEVALUE, TRUEVALUE)
#define AVX512F_INTS                         __m512i

#define AVX512F_MASKZERO_COMPRESS_INT(MASK, X)   _mm512_maskz_compress_epi32(MASK, X)

  //Bitwise ops work on masks
#define AVX512F_BITWISE_AND(X,Y)              _mm512_kand(X,Y)
#define AVX512F_BITWISE_OR(X,Y)               _mm512_kor(X,Y)
#define AVX512F_XOR_FLOATS(X,Y)               _mm512_kxor(X,Y)
#define AVX512F_AND_NOT(X,Y)                  _mm512_kandnot((X),(Y))  //~X & Y


#ifndef DOUBLE_PREC

#define DOUBLE                               float  
#define AVX512F_NVEC                         16    
#define AVX512F_FLOATS                       __m512
#define AVX512F_MASK                         __mmask8
  
#define AVX512F_LOAD_FLOATS_UNALIGNED(X)     _mm512_loadu_ps(X)
#define AVX512F_LOAD_FLOATS_ALIGNED(X)       _mm512_load_ps(X)
#define AVX512F_MULTIPLY_FLOATS(X,Y)         _mm512_mul_ps(X,Y)
#define AVX512F_DIVIDE_FLOATS(X,Y)           _mm512_div_ps(X,Y)
#define AVX512F_SUBTRACT_FLOATS(X,Y)         _mm512_sub_ps(X,Y)
#define AVX512F_ADD_FLOATS(X,Y)              _mm512_add_ps(X,Y)
#define AVX512F_SQRT_FLOAT(X)                _mm512_sqrt_ps(X)
#define AVX512F_SVML_SQRT_FLOAT(X)           _mm512_svml_sqrt_ps(X)
#define AVX512F_TRUNCATE_FLOAT_TO_INT(X)     _mm512_cvttps_epi32(X)
#define AVX512F_STORE_FLOATS_TO_MEMORY(X,Y)  _mm512_storeu_ps(X,Y)
#define AVX512F_SQUARE_FLOAT(X)              _mm512_mul_ps(X,X)
#define AVX512F_LOG_FLOAT(X)                 _mm512_log_ps(X)
#define AVX512F_LOG10_FLOAT(X)                 _mm512_log10_ps(X)
#define AVX512F_LOG2_FLOAT(X)                _mm512_log2_ps(X)
#define AVX512F_RECIPROCAL_FLOATS(X)         _mm512_rcp_ps(X)

#define AVX512F_BROADCAST_FLOAT(X)           _mm512_broadcast_ss(X);
#define AVX512F_SET_FLOAT(X)                 _mm512_set1_ps(X);



#define AVX512F_MASKZERO_COMPRESS_FLOAT(MASK, X)   _mm512_maskz_compress_ps(MASK, X)

// X OP Y
#define AVX512F_COMPARE_FLOATS(X,Y,OP)        _mm512_cmp_ps_mask(X,Y,OP)

//MoveMask
#define AVX512F_TEST_COMPARISON(X)            _mm512_movemask_ps(X)

#define AVX512F_BLEND_FLOATS_WITH_MASK(MASK, FALSEVALUE,TRUEVALUE) _mm512_mask_blend_ps(MASK, FALSEVALUE, TRUEVALUE)
#define AVX512F_MASKSTORE_FLOATS(dest, mask, source)   _mm512_maskstore_ps(dest, mask, source)

//Trig
#ifdef  __INTEL_COMPILER
#define AVX512F_ARC_COSINE(X, order)                 _mm512_acos_ps(X)
#else
    //Other compilers do not have the vectorized arc-cosine
#define AVX512F_ARC_COSINE(X, order)                  inv_cosine_avx(X, order)
#endif

    //Max
#define AVX512F_MAX_FLOATS(X,Y)               _mm512_max_ps(X,Y)

    //Casting (does not actual convert between types)
#define AVX512F_CAST_FLOAT_TO_INT(X)          _mm512_castps_si512(X)
#define AVX512F_CAST_INT_TO_FLOAT(X)          _mm512_castsi512_ps(X)

    //Streaming store
#define AVX512F_STREAMING_STORE_FLOATS(X,Y)   _mm512_stream_ps(X,Y)
#define AVX512F_STREAMING_STORE_INTS(X,Y)     _mm512_stream_si512(X,Y)

#else //DOUBLE PRECISION CALCULATIONS
  
#define DOUBLE                           double
#define AVX512F_NVEC                         4    
#define AVX512F_FLOATS                       __m512d
#define AVX512F_MASK                         __mmask16

#define AVX512F_LOAD_FLOATS_UNALIGNED(X)     _mm512_loadu_pd(X)
#define AVX512F_LOAD_FLOATS_ALIGNED(X)       _mm512_load_pd(X)
#define AVX512F_MULTIPLY_FLOATS(X,Y)         _mm512_mul_pd(X,Y)
#define AVX512F_DIVIDE_FLOATS(X,Y)           _mm512_div_pd(X,Y)
#define AVX512F_SUBTRACT_FLOATS(X,Y)         _mm512_sub_pd(X,Y)
#define AVX512F_ADD_FLOATS(X,Y)              _mm512_add_pd(X,Y)
#define AVX512F_SQRT_FLOAT(X)                _mm512_sqrt_pd(X)
#define AVX512F_SVML_SQRT_FLOAT(X)           _mm512_svml_sqrt_pd(X)
#define AVX512F_TRUNCATE_FLOAT_TO_INT(X)     _mm512_cvttpd_epi32(X)
#define AVX512F_STORE_FLOATS_TO_MEMORY(X,Y)  _mm512_storeu_pd(X,Y)
#define AVX512F_SQUARE_FLOAT(X)              _mm512_mul_pd(X,X)
#define AVX512F_LOG_FLOAT(X)                 _mm512_log_pd(X)
#define AVX512F_LOG2_FLOAT(X)                _mm512_log2_pd(X)
#define AVX512F_LOG10_FLOAT(X)                _mm512_log10_pd(X)
#define AVX512F_RECIPROCAL_FLOATS(X)         _mm512_rcp_pd(X)


#define AVX512F_MASKZERO_COMPRESS_FLOAT(MASK, X)   _mm512_maskz_compress_pd(MASK, X)

    // X OP Y
#define AVX512F_COMPARE_FLOATS(X,Y,OP)        _mm512_cmp_pd_mask(X,Y,OP)

#define AVX512F_BROADCAST_FLOAT(X)            _mm512_broadcast_sd(X);
#define AVX512F_SET_FLOAT(X)                  _mm512_set1_pd(X);
//MoveMask
#define AVX512F_TEST_COMPARISON(X)            _mm512_movemask_pd(X)


#define AVX512F_BLEND_FLOATS_WITH_MASK(MASK, FALSEVALUE,TRUEVALUE) _mm512_mask_blend_pd(MASK, FALSEVALUE, TRUEVALUE)
#define AVX512F_MASKSTORE_FLOATS(dest, mask, source)   _mm512_maskstore_pd(dest, mask, source)

//Trig
#ifdef  __INTEL_COMPILER
#define AVX512F_ARC_COSINE(X, order)                 _mm512_acos_pd(X)
#else
#define AVX512F_ARC_COSINE(X, order)                  inv_cosine_avx(X, order)
#endif

    //Max
#define AVX512F_MAX_FLOATS(X,Y)               _mm512_max_pd(X,Y)

    //Casting (does not actual convert between types)
#define AVX512F_CAST_FLOAT_TO_INT(X)          _mm512_castpd_si512(X)
#define AVX512F_CAST_INT_TO_FLOAT(X)          _mm512_castsi512_pd(X)

    //Streaming store
#define AVX512F_STREAMING_STORE_FLOATS(X,Y)   _mm512_stream_pd(X,Y)
#define AVX512F_STREAMING_STORE_INTS(X,Y)     _mm_stream_si128(X,Y)

#endif //DOUBLE_PREC

#ifndef  __INTEL_COMPILER
#include "fast_acos.h"
    
static inline AVX512F_FLOATS inv_cosine_avx(const AVX512F_FLOATS X, const int order)
{
    union cos{
        AVX512F_FLOATS m;
        DOUBLE x[AVX512F_NVEC];
    };
    union cos union_costheta;
    union cos union_returnvalue;
    union_costheta.m = X;
    const DOUBLE minus_one = (DOUBLE) -1.0;
    const DOUBLE one = (DOUBLE) 1.0;

    //Force everything to be in range [0,1]
    for(int ii=0;ii<AVX512F_NVEC;ii++) {
        const DOUBLE costheta = union_costheta.x[ii];
        union_costheta.x[ii] = costheta <= minus_one ? minus_one:costheta;
        union_costheta.x[ii] = costheta >= one ? one:costheta;
    }
    
    if(order == 0) {
        for(int ii=0;ii<AVX512F_NVEC;ii++) {
            const DOUBLE costheta = union_costheta.x[ii];
            union_returnvalue.x[ii] = ACOS(costheta);
        }
    } else {
        //fast acos
        /*Taken from associated C++ code in http://www.geometrictools.com/GTEngine/Include/Mathematics/GteACosEstimate.h*/
        for(int ii=0;ii<AVX512F_NVEC;ii++) {
            union_returnvalue.x[ii] = FAST_ACOS(union_costheta.x[ii]);
        }
    }
    return union_returnvalue.m;
  }


#endif


#ifdef __cplusplus
}
#endif
