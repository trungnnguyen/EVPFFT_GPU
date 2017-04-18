set -x

#FFT_FLAGS=-DUSE_FOURN
#FFT_FLAGS=-DUSE_FFTW
#FFT_LIBS=-lfftw3f
FFT_FLAGS=-DUSE_CUFFT
FFT_LIBS="-Mcuda" # -Mcudalib=cufft"
ACC_FLAGS=-ta=tesla

pgf90 -DGPU -Mpreprocess -Minfo=accel $xACC_FLAGS -fast -tp=p7 -c evpnew10.for
pgcc $FFT_FLAGS -Minfo=accel,all $ACC_FLAGS -fast -tp=p7 -c fourn.c
pgf90 $ACC_FLAGS evpnew10.o fourn.o -lrt -o evpnew10 -Mcudalib=cufft

