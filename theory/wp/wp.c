/* File: wp.c */
/*
  This file is a part of the Corrfunc package
  Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
  License: MIT LICENSE. See LICENSE file under the top-level
  directory at https://github.com/manodeep/Corrfunc/
*/

/* PROGRAM wp
   --- wp boxsize file format binfile pimax [Nthreads] > wpfile
   --- Measure the projected auto-correlation function wp(rp) for a single periodic box
   * boxsize      = BoxSize (in same units as X/Y/Z of the data)
   * file         = name of first data file
   * format       = format of first data file  (a=ascii, c=csv, f=fast-food)
   * binfile      = name of ascii file containing the r-bins (rmin rmax for each bin)
   * pimax        = maximum line-of-sight-separation
   * numthreads   = number of threads to use (only if USE_OMP is enabled)
   > wpfile         = name of output file. Contains <wp [rpavg=0.0] rmin rmax npairs>
   ----------------------------------------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <inttypes.h>

#include "defs.h" //for ADD_DIFF_TIME
#include "function_precision.h" //definition of DOUBLE

#include "countpairs_wp.h" //function proto-type for countpairs
#include "io.h" //function proto-type for file input
#include "utils.h" //general utilities

void Printhelp(void);

int main(int argc, char *argv[])
{

    /*---Arguments-------------------------*/
    double boxsize;
    char *file=NULL,*fileformat=NULL;
    char *binfile=NULL;
    DOUBLE pimax ;

    /*---Data-variables--------------------*/
    int64_t ND1=0;
    DOUBLE *x1=NULL,*y1=NULL,*z1=NULL;


    /*---Corrfunc-variables----------------*/
#if !(defined(USE_OMP) && defined(_OPENMP))
    const int nthreads = 1;
    const char argnames[][30]={"boxsize","file","format","binfile","pimax"};
#else
    int nthreads=2;
    const char argnames[][30]={"boxsize","file","format","binfile","pimax","Nthreads"};
#endif
    int nargs=sizeof(argnames)/(sizeof(char)*30);

    struct timeval t_end,t_start,t0,t1;
    double read_time=0.0;
    gettimeofday(&t_start,NULL);

    /*---Read-arguments-----------------------------------*/
    if(argc< (nargs+1)) {
        Printhelp() ;
        fprintf(stderr,"\nFound: %d parameters\n ",argc-1);
        int i;
        for(i=1;i<argc;i++) {
            if(i <= nargs)
                fprintf(stderr,"\t\t %s = `%s' \n",argnames[i-1],argv[i]);
            else
                fprintf(stderr,"\t\t <> = `%s' \n",argv[i]);
        }
        if(i <= nargs) {
            fprintf(stderr,"\nMissing required parameters \n");
            for(i=argc;i<=nargs;i++)
                fprintf(stderr,"\t\t %s = `?'\n",argnames[i-1]);
        }
        return EXIT_FAILURE;
    }
    boxsize=atof(argv[1]);
    file=argv[2];
    fileformat=argv[3];
    binfile=argv[4];

    pimax=40.0;

#ifdef DOUBLE_PREC
    sscanf(argv[5],"%lf",&pimax) ;
#else
    sscanf(argv[5],"%f",&pimax) ;
#endif


#if defined(USE_OMP) && defined(_OPENMP)
    nthreads=atoi(argv[6]);
    if( nthreads < 1 ) {
      fprintf(stderr,"Number of threads = %d must be >=1 \n", nthreads);
      return EXIT_FAILURE;
    }
#endif

    fprintf(stderr,"Running `%s' with the parameters \n",argv[0]);
    fprintf(stderr,"\n\t\t -------------------------------------\n");
    for(int i=1;i<argc;i++) {
        if(i <= nargs) {
            fprintf(stderr,"\t\t %-10s = %s \n",argnames[i-1],argv[i]);
        }  else {
            fprintf(stderr,"\t\t <> = `%s' \n",argv[i]);
        }
    }
    fprintf(stderr,"\t\t -------------------------------------\n");


    gettimeofday(&t0,NULL);
    /*---Read-data1-file----------------------------------*/
    ND1=read_positions(file,fileformat,sizeof(DOUBLE), 3, &x1, &y1, &z1);
    gettimeofday(&t1,NULL);
    read_time += ADD_DIFF_TIME(t0,t1);

    //check that theee positions are within limits
    for(int i=0;i<ND1;i++) {
        assert(x1[i] >= 0.0 && x1[i] <= boxsize && "xpos is within limits [0, boxsize]");
        assert(y1[i] >= 0.0 && y1[i] <= boxsize && "ypos is within limits [0, boxsize]");
        assert(z1[i] >= 0.0 && z1[i] <= boxsize && "zpos is within limits [0, boxsize]");
    }

    /*---Count-pairs--------------------------------------*/
    gettimeofday(&t0,NULL);
    struct config_options options = get_config_options();
    results_countpairs_wp results;
    int status = countpairs_wp(ND1, x1, y1, z1,
                               boxsize,
                               nthreads,
                               binfile,
                               pimax,
                               &results,
                               &options,
                               NULL);
    free(x1);free(y1);free(z1);
    if(status != EXIT_SUCCESS) {
        return status;
    }
    
    gettimeofday(&t1,NULL);
    double pair_time = ADD_DIFF_TIME(t0,t1);

    //Output the results
    /* Note: we discard the first bin, to mimic the fact that close pairs
     * are disregarded in SDSS data.
     */
    double rlow=results.rupp[0];
    for(int i=1;i<results.nbin;++i) {
        fprintf(stdout,"%e\t%e\t%e\t%12"PRIu64"\t%e\n",rlow,results.rupp[i],results.rpavg[i],results.npairs[i],results.wp[i]);
        rlow=results.rupp[i];
    }

    //free the memory in the results struct
    free_results_wp(&results);

    gettimeofday(&t_end,NULL);
    fprintf(stderr,"wp> Done -  ND1=%12"PRId64". Time taken = %6.2lf seconds. read-in time = %6.2lf seconds pair-counting time = %6.2lf sec\n",
            ND1,ADD_DIFF_TIME(t_start,t_end),read_time,pair_time);
    return EXIT_SUCCESS;
}

/*---Print-help-information---------------------------*/
void Printhelp(void)
{
    fprintf(stderr,"=========================================================================\n") ;
    fprintf(stderr,"   --- wp boxsize file format binfile pimax [Nthreads] > wpfile\n") ;
    fprintf(stderr,"   --- Measure the projected auto-correlation function wp(rp) for a single periodic box\n") ;
    fprintf(stderr,"     * boxsize      = BoxSize (in same units as X/Y/Z of the data)\n") ;
    fprintf(stderr,"     * file         = name of data file\n") ;
    fprintf(stderr,"     * format       = format of data file  (a=ascii, c=csv, f=fast-food)\n") ;
    fprintf(stderr,"     * binfile       = name of ascii file containing the r-bins (rmin rmax for each bin)\n") ;
    fprintf(stderr,"     * pimax         = maximum line-of-sight-separation\n") ;
#if defined(USE_OMP) && defined(_OPENMP)
    fprintf(stderr,"     * numthreads    = number of threads to use\n");
#endif

#ifdef OUTPUT_RPAVG
    fprintf(stderr,"     > wpfile        = name of output file. Contains <wp  rpavg  rmin rmax npairs>\n") ;
#else
    fprintf(stderr,"     > wpfile        = name of output file. Contains <wp [rpavg=0.0] rmin rmax npairs>\n") ;
#endif

    fprintf(stderr,"\n\tCompile options: \n");
#ifdef PERIODIC
    fprintf(stderr,"Periodic = True\n");
#else
    fprintf(stderr,"Periodic = False\n");
#endif

#ifdef OUTPUT_RPAVG
    fprintf(stderr,"Output RPAVG = True\n");
#else
    fprintf(stderr,"Output RPAVG = False\n");
#endif

#ifdef DOUBLE_PREC
    fprintf(stderr,"Precision = double\n");
#else
    fprintf(stderr,"Precision = float\n");
#endif

#if defined(USE_OMP) && defined(_OPENMP)
    fprintf(stderr,"Use OMP = True\n");
#else
    fprintf(stderr,"Use OMP = False\n");
#endif

    fprintf(stderr,"=========================================================================\n") ;


}