
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#ifdef USE_CUFFT
#include <cufft.h> 
#endif
#ifdef USE_FFTW
#include <fftw3.h>
#endif

static inline double
WTime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec / 1e6;
}

// ----------------------------------------------------------------------
// FOURN

void fourn(float *, int nn[4], int, int);  //Fourier Transform

static void
fft_forward_fourn(float *data, int nn[3])
{
  fourn(data - 1, nn - 1, 3, -1);
}

static void
fft_backward_fourn(float *data, int nn[3])
{
  fourn(data - 1, nn - 1, 3, 1);
}

// ----------------------------------------------------------------------
// FFTW

#ifdef USE_FFTW

static void
fft_forward_fftw(float *data, int nn[3])
{
  static fftwf_plan plan;
  if (!plan) {
    plan = fftwf_plan_dft_3d(nn[0],
			     nn[1],
			     nn[2],
			     (fftwf_complex *) data,
			     (fftwf_complex *) data,
			     FFTW_FORWARD,
			     FFTW_ESTIMATE);
  }

  fftwf_execute_dft(plan, (fftwf_complex *) data,
		    (fftwf_complex *) data);
}

static void
fft_backward_fftw(float *data, int nn[3])
{
  static fftwf_plan plan;
  if (!plan) {
    plan = fftwf_plan_dft_3d(nn[0],
			     nn[1],
			     nn[2],
			     (fftwf_complex *) data,
			     (fftwf_complex *) data,
			     FFTW_BACKWARD,
			     FFTW_ESTIMATE);
  }

  fftwf_execute_dft(plan, (fftwf_complex *) data,
		    (fftwf_complex *) data);
}

#endif

// ----------------------------------------------------------------------
// CUFFT

#ifdef USE_CUFFT

static void
fft_forward_cufft(float *data, int nn[3], int batch)
{
  int stride = 2 * nn[0] * nn[1] * nn[2];
  int rc, i;

  static cufftHandle planc2c;
  if (!planc2c) {
    cufftPlan3d(&planc2c, nn[0], nn[1], nn[2], CUFFT_C2C);
  }

#pragma acc data copy(data[0:batch*stride])
  {
    for (i = 0; i < batch; i++) {
//      printf("data1 %p\n", data);
#pragma acc host_data use_device(data)
      {
//      printf("data2 %p\n", data);
      rc = cufftExecC2C(planc2c, (cufftComplex *)(data + i * stride),
			(cufftComplex *)(data + i * stride),
			CUFFT_FORWARD);
      assert(rc == CUFFT_SUCCESS);
      }
    }
  }
}

static void
fft_backward_cufft(float *data, int nn[3], int batch)
{
  int stride = 2 * nn[0] * nn[1] * nn[2];
  int rc, i;

  static cufftHandle planc2c;
  if (!planc2c) {
    cufftPlan3d(&planc2c, nn[0], nn[1], nn[2], CUFFT_C2C);
  }

#pragma acc data copy(data[0:batch*stride])
  {
    for (i = 0; i < batch; i++) {
#pragma acc host_data use_device(data)
      rc = cufftExecC2C(planc2c, (cufftComplex *)(data + i * stride),
			(cufftComplex *)(data + i * stride),
			CUFFT_INVERSE);
      assert(rc == CUFFT_SUCCESS);
    }
  }
}
#endif

// ----------------------------------------------------------------------

static void
fft_forward(float *data, int nn[3], int batch)
{
  int stride = 2 * nn[0] * nn[1] * nn[2];
  int i;

  static double acc_time = 0;
  static int acc_n = 0;
  double tmstart = WTime();
  
#if defined(USE_FOURN)
  for (i = 0; i < batch; i++) {
    fft_forward_fourn(data + i * stride, nn);
  }
#elif defined(USE_FFTW)
  for (i = 0; i < batch; i++) {
    fft_forward_fftw(data + i * stride, nn);
  }
#elif defined(USE_CUFFT)
  fft_forward_cufft(data, nn, batch);
#else
#error I do not know which FFT to use  
#endif

  double now = WTime();
  acc_time += now - tmstart;
  acc_n++;

  if (0 && acc_n % 10 == 0) {
    printf("avg FFT FORWARD time : called %d times %g total time %g\n", acc_n, acc_time / acc_n, acc_time);
  }
}

static void
fft_backward(float *data, int nn[3], int batch)
{
  int stride = 2 * nn[0] * nn[1] * nn[2];
  int i;

  static double acc_time = 0;
  static int acc_n = 0;
  double tmstart = WTime();

#if defined(USE_FOURN)
  for (i = 0; i < batch; i++) {
    fft_backward_fourn(data + i * stride, nn);
  }
#elif defined(USE_FFTW)
  for (i = 0; i < batch; i++) {
    fft_backward_fftw(data + i * stride, nn);
  }
#elif defined(USE_CUFFT)
  fft_backward_cufft(data, nn, batch);
#else
#error I do not know which FFT to use  
#endif

  double now = WTime();
  acc_time += now - tmstart;
  acc_n++;

  if (0 && acc_n % 10 == 0) {
    printf("avg FFT BACKWARD time : called %d times %g total time %g\n", acc_n, acc_time / acc_n, acc_time);
  }
}

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr

void fourn(float data[], int nn[], int ndim, int isign)
{
	int idim;
	unsigned long i1,i2,i3,i2rev,i3rev,ip1,ip2,ip3,ifp1,ifp2;
	unsigned long ibit,k1,k2,n,nprev,nrem,ntot;
	float tempi,tempr;
	double theta,wi,wpi,wpr,wr,wtemp;

	for (ntot=1,idim=1;idim<=ndim;idim++)
		ntot *= nn[idim];
	nprev=1;
	for (idim=ndim;idim>=1;idim--) {
		n=nn[idim];
		nrem=ntot/(n*nprev);
		ip1=nprev << 1;
		ip2=ip1*n;
		ip3=ip2*nrem;
		i2rev=1;
		/*printf("%u, %u, %u, %u\n", ip1, ip2, ip3, nprev);*/
		for (i2=1;i2<=ip2;i2+=ip1) {
			if (i2 < i2rev) {
			  /*printf("%u, %u, %u, %u\n", i1, i2, i2rev, ibit);*/
				for (i1=i2;i1<=i2+ip1-2;i1+=2) {
				  /*printf("i1 %u, %u\n", i1, i2);*/
					for (i3=i1;i3<=ip3;i3+=ip2) {
						i3rev=i2rev+i3-i2;
						SWAP(data[i3],data[i3rev]);
						SWAP(data[i3+1],data[i3rev+1]);
					}
				}
			}
			ibit=ip2 >> 1;
			/*printf("ibit %u\n", ibit);*/
			while (ibit >= ip1 && i2rev > ibit) {
				i2rev -= ibit;
				ibit >>= 1;
			}
			i2rev += ibit;
		}
		ifp1=ip1;
		while (ifp1 < ip2) {
			ifp2=ifp1 << 1;
			theta=isign*6.28318530717959/(ifp2/ip1);
			wtemp=sin(0.5*theta);
			wpr = -2.0*wtemp*wtemp;
			wpi=sin(theta);
			wr=1.0;
			wi=0.0;
			for (i3=1;i3<=ifp1;i3+=ip1) {
				for (i1=i3;i1<=i3+ip1-2;i1+=2) {
					for (i2=i1;i2<=ip3;i2+=ifp2) {
						k1=i2;
						k2=k1+ifp1;
						tempr=(float)wr*data[k2]-(float)wi*data[k2+1];
						tempi=(float)wr*data[k2+1]+(float)wi*data[k2];
						data[k2]=data[k1]-tempr;
						data[k2+1]=data[k1+1]-tempi;
						data[k1] += tempr;
						data[k1+1] += tempi;
					}
				}
				wr=(wtemp=wr)*wpr-wi*wpi+wr;
				wi=wi*wpr+wtemp*wpi+wi;
			}
			ifp1=ifp2;
		}
		nprev *= n;
		/*printf("nprev %u, %u\n", nprev, n);*/
	}
	//        printf("data %g %g\n", data[1], data[2]);
}

void c_fourn_(float data[], int nn[], int *ndim, int *isign)
{
  assert(*ndim == 3);
  if (*isign == -1) {
    fft_forward(data, nn, 1);
  } else {
    fft_backward(data, nn, 1);
  }
  //  fourn(data - 1, nn - 1, *ndim, *isign);
}
