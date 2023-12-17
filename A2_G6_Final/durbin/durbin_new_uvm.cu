// Includes.
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>


// Macros.
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define gpuErrchk(ans)                      \
{                                           \
    gpuAssert((ans), __FILE__, __LINE__);   \
}

extern "C"
{
  #include "utils.h"
}

// Dataset size.
#ifndef N
  #define N 10000
#endif

// Block size.
#ifndef BLOCK_SIZE
  #define BLOCK_SIZE 32
#endif

// Data type.
#ifndef DATA_T
  #define DATA_T double
#endif


// Utility per scambiare tra loro due puntatori (y_old/y_new).
void swapPointers(DATA_T *__restrict__(&ptr1), DATA_T *__restrict__(&ptr2)) {
  DATA_T *__restrict__ temp = ptr1;
  ptr1 = ptr2;
  ptr2 = temp;
}


// Host kernel.
void kernel_durbin_host(DATA_T * r, DATA_T * out)
{
  int i, k;
  DATA_T sum, alpha, beta;
  DATA_T y[2][N];
  alpha = r[0];
  beta = 1;
  y[0][0] = r[0];

  for (k = 1; k < N; k++)
  {
    beta = beta - alpha * alpha * beta;
    sum = r[k];

    for (i = 0; i <= k - 1; i++)
      sum += r[k - i - 1] * y[(k - 1) % 2][i];
    
    alpha = -sum * beta;

    for (i = 0; i <= k - 1; i++)
      y[k % 2][i] = y[(k - 1) % 2][i] + alpha * y[(k - 1) % 2][k - i - 1];
    
    y[k % 2][k] = alpha;
  }

  for (i = 0; i < N; i++)
    out[i] = y[(N - 1) % 2][i];
}


// Device kernel.
// Primo kernel -> calcolo delle somme parziali e successiva reduction su sum.
__global__ void first_kernel(
  DATA_T *__restrict__ y,
  DATA_T *__restrict__ r,
  int k,
  DATA_T *__restrict__ sum)
{
  __shared__ DATA_T partialsum[BLOCK_SIZE];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Caricamento dei dati in memoria condivisa
  if (i < k)
    partialsum[tid] = r[k - i - 1] * y[i];
  else
    partialsum[tid] = 0;

  __syncthreads();

  // Riduzione a blocchi
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
      partialsum[tid] += partialsum[tid + stride];
      
    __syncthreads();
  }

  // Il thread 0 di ciascun blocco aggiorna il valore globale nel device
  if (tid == 0)
    atomicAdd(sum, partialsum[0]);
}

// Secondo kernel -> calcolo del nuovo y in stile saxpy.
__global__ void second_kernel(
  DATA_T *__restrict__ y_old,
  DATA_T *__restrict__ y_new,
  int k,
  DATA_T *__restrict__ alpha)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < k)
    y_new[i] = y_old[i] + alpha[0] * y_old[k - i - 1];

  if (i == k)
    y_new[i] = alpha[0];
}

// Funzione chiamante dei kernel. Replica il kernel di Durbin.
void kernel_durbin_device(
  DATA_T *__restrict__ y_old,
  DATA_T *__restrict__ y_new,
  DATA_T *__restrict__ r,
  DATA_T *__restrict__ alpha,
  DATA_T *__restrict__ beta,
  DATA_T *__restrict__ sum)
{
  int k;
  int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (k = 1; k < N; k++)
  {
    // Calcolo beta e sum.
    // sum[0] = r[k];
    cudaMemcpy(sum, &r[k], sizeof(DATA_T), cudaMemcpyDeviceToDevice);
    beta[0] = beta[0] - alpha[0] * alpha[0] * beta[0];
    first_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(y_old, r, k, sum);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // Calcolo del nuovo alpha.
    alpha[0] = -sum[0] * beta[0];

    // Calcolo del nuovo y.
    second_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(y_old, y_new, k, alpha);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // Scambio degli y.
    swapPointers(y_old, y_new);
  }
}


int main(int argc, char **argv)
{
  // Default data structures.
  int iret = 0;
  struct timespec rt[2];
  DATA_T wt;

  // Algorithm data structures.
  DATA_T *h_r, *h_out, *d_out;

  if (NULL == (h_r = (DATA_T *)malloc(sizeof(*h_r) * N)))
  {
    printf("error: memory allocation for 'h_r'\n");
    iret = -1;
  }
  if (NULL == (h_out = (DATA_T *)malloc(sizeof(*h_out) * N)))
  {
    printf("error: memory allocation for 'h_out'\n");
    iret = -1;
  }
  if (NULL == (d_out = (DATA_T *)malloc(sizeof(*d_out) * N)))
  {
    printf("error: memory allocation for 'd_out'\n");
    iret = -1;
  }

  // Device data structures.
  DATA_T *d_r, *y_old, *y_new;

  // Device mallocs.
  gpuErrchk(cudaMalloc((void **)&d_r, sizeof(DATA_T) * N));
  gpuErrchk(cudaMalloc((void **)&y_old, sizeof(DATA_T) * N));
  gpuErrchk(cudaMalloc((void **)&y_new, sizeof(DATA_T) * N));

  // Return if any error occurred in mallocs.
  if (0 != iret)
  {
    free(h_r);
    free(h_out);
    free(d_out);
    gpuErrchk(cudaFree(d_r));
    gpuErrchk(cudaFree(y_old));
    gpuErrchk(cudaFree(y_new));
    exit(EXIT_FAILURE);
  }

  // Init data.
  int i;

  #pragma omp parallel for
  for (i = 0; i < N; i++)
    h_r[i] = (DATA_T)(i + 1) / N / 4.0;

  // Test host.
  // Start timer.
  clock_gettime(CLOCK_REALTIME, rt + 0);

  // Function kernel durbin call (host).
  kernel_durbin_host(h_r, h_out);

  // Stop timer.
  clock_gettime(CLOCK_REALTIME, rt + 1);

  // Print results.
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf("Durbin (Host) : %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * N * N * N / (1.0e9 * wt));

  // Test device.
  // Start timer.
  clock_gettime(CLOCK_REALTIME, rt + 0);

  // Memcopies.
  // Device's array r.
  gpuErrchk(cudaMemcpy(d_r, h_r, sizeof(DATA_T) * N, cudaMemcpyHostToDevice));
  // y_old[0] = r[0].
  gpuErrchk(cudaMemcpy(y_old, d_r, sizeof(DATA_T), cudaMemcpyDeviceToDevice));

  // UVM mallocs.
  DATA_T *alpha_d, *beta_d, *sum_d;
  gpuErrchk(cudaMallocManaged((void **)&alpha_d, sizeof(DATA_T)));
  gpuErrchk(cudaMallocManaged((void **)&beta_d, sizeof(DATA_T)));
  gpuErrchk(cudaMallocManaged((void **)&sum_d, sizeof(DATA_T)));

  // UVM inits.
  alpha_d[0] = h_r[0];
  beta_d[0] = 1;
  gpuErrchk(cudaMemcpy(y_old, h_r, sizeof(DATA_T), cudaMemcpyHostToDevice));

  // Function kernel durbin call (device).
  kernel_durbin_device(y_old, y_new, d_r, alpha_d, beta_d, sum_d);

  // out = y_new dell'ultima iterazione di durbin (viene swappato -> quindi y_old).
  gpuErrchk(cudaMemcpy(d_out, y_new, sizeof(DATA_T) * N, cudaMemcpyDeviceToHost));

  // Debug.
  // for (i = 0; i < N; i++)
  //   printf("out[%d] = %f\n", i, d_out[i]);

  // Stop timer.
  clock_gettime(CLOCK_REALTIME, rt + 1);

  // Print results.
  wt = (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  printf("Durbin (GPU): %9.3f sec %9.1f GFLOPS\n", wt, 2.0 * N * N * N / (1.0e9 * wt));

  // Frees.
  free(h_r);
  free(h_out);
  free(d_out);
  gpuErrchk(cudaFree(d_r));
  gpuErrchk(cudaFree(y_old));
  gpuErrchk(cudaFree(y_new));
  gpuErrchk(cudaFree(alpha_d));
  gpuErrchk(cudaFree(beta_d));
  gpuErrchk(cudaFree(sum_d));

  return 0;
}
