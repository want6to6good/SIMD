#include<iostream>
#include<windows.h>
#include<nmmintrin.h>  // SSE 
#include<immintrin.h>  // AVX

using namespace std;


int n;
float** b;

void init(float** a)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			a[i][j] = b[i][j];
}

void init_()
{
	b = new float* [n];
	for (int i = 0; i < n; i++)
	{
		b[i] = new float[n];
		for (int j = 0; j < n; j++)
			b[i][j] = rand() % 10;
	}
}

float** common()
{
	float** m = new float* [n];
	for (int i = 0; i < n; i++)
		m[i] = new float[n];
	init(m);
	return m;
}

float** special() 
{
	float** m = (float**)_aligned_malloc(32 * n * sizeof(float**), 32);
	for (int i = 0; i < n; i++)
		m[i] = (float*)_aligned_malloc(32 * n * sizeof(float*), 32);
	init(m);
	return m;
}

void gauss(float** a)// 标准的高斯消去算法
{  
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
				a[i][j] -= a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}
}

void gauss_cache(float** a) // Cache优化版本
{  
	float t1, t2;  
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
		for (int j = k + 1; j < n; j++)
			a[k][j] = a[k][j] / t1;
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)				
				a[i][j] -= t2 * a[k][j];
			a[i][k] = 0;
		}
	}
}

void SSE_U(float** a)// 未对齐
{
	__m128 va, vt, vaik, vakj, vaij, vx;
	for (int k = 0; k < n; k++)
	{
		vt = _mm_set1_ps(a[k][k]);  
		int j = k + 1;
		for (; j + 4 < n; j += 4)
		{
			va = _mm_loadu_ps(&a[k][j]);
			va = _mm_div_ps(va, vt);
			_mm_storeu_ps(&a[k][j], va);
		}
		for (; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm_set1_ps(a[i][k]);
			for (j = k + 1; j + 4 < n; j += 4)
			{
				vakj = _mm_loadu_ps(&a[k][j]);
				vaij = _mm_loadu_ps(&a[i][j]);
				vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_storeu_ps(&a[i][j], vaij);
			}
			for (; j < n; j++)
				a[i][j] -= a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}
}

void AVX_U(float** a) {  // 未对齐
	__m256 va, vt, vaik, vakj, vaij, vx;
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(a[k][k]);
		int j = k + 1;
		for (; j + 8 < n; j += 8)
		{
			va = _mm256_loadu_ps(&a[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(&a[k][j], va);
		}
		for (; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm256_set1_ps(a[i][k]);
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&a[k][j]);
				vaij = _mm256_loadu_ps(&a[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&a[i][j], vaij);
			}
			for (; j < n; j++)
				a[i][j] -= a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}
}

void SSE(float** a) // 对齐
{  
	__m128 va, vt, vaik, vakj, vaij, vx;
	for (int k = 0; k < n; k++)
	{
		vt = _mm_set1_ps(a[k][k]); 
		int j = k+1;
		for (; j + 4 < n; j += 4)
		{
			va = _mm_load_ps(&a[k][j]);
			va = _mm_div_ps(va, vt);
			_mm_store_ps(&a[k][j], va);
		}
		for (; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm_set1_ps(a[i][k]);
			for (j = k + 1; j + 4 < n; j += 4)
			{
				vakj = _mm_load_ps(&a[k][j]);
				vaij = _mm_load_ps(&a[i][j]);
				vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_store_ps(&a[i][j], vaij);
			}
			for (; j < n; j++)
				a[i][j] -= a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}
}

void AVX(float** a) // 对齐
{  
	__m256 va, vt, vaik, vakj, vaij, vx;
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(a[k][k]);
		int j = k + 1;
		for (; j + 8 < n; j += 8)
		{
			va = _mm256_load_ps(&a[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_store_ps(&a[k][j], va);
		}
		for (; j < n; j++)
			a[k][j] = a[k][j] / a[k][k]; 
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm256_set1_ps(a[i][k]);
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_load_ps(&a[k][j]);
				vaij = _mm256_load_ps(&a[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_store_ps(&a[i][j], vaij);
			}
			for (; j < n; j++)
				a[i][j] -= a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}
}


int main() {
	cin >> n;
	cout << "问题规模为" << n <<"*"<<n<<"的矩阵"<<endl;

	init_();

	long long h1, h2, h3, h4, h5, h6, t1, t2, t3, t4, t5, t6, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	
	cout << endl << endl;


	float** m1 = common();
	QueryPerformanceCounter((LARGE_INTEGER*)&h1);
	gauss(m1);
	QueryPerformanceCounter((LARGE_INTEGER*)&t1);
	cout << "gauss普通版本: " << (t1 - h1) * 1000.0 / freq<< "ms" << endl;


	float** m2 = common();
	QueryPerformanceCounter((LARGE_INTEGER*)&h2);
	gauss_cache(m2);
	QueryPerformanceCounter((LARGE_INTEGER*)&t2);
	cout << "gauss cache优化版本: " << (t2 - h2) * 1000.0 / freq<< "ms" << endl;


	float** m3 = special();
	QueryPerformanceCounter((LARGE_INTEGER*)&h3);
	SSE(m3);
	QueryPerformanceCounter((LARGE_INTEGER*)&t3);
	cout << "SSE对齐: " << (t3 - h3) * 1000.0 / freq<< "ms" << endl;


	float** m4 = common();
	QueryPerformanceCounter((LARGE_INTEGER*)&h4);
	SSE_U(m4);
	QueryPerformanceCounter((LARGE_INTEGER*)&t4);
	cout << "SSE不对齐: " << (t4 - h4) * 1000.0 / freq<< "ms" << endl;


	float** m5 = special();
	QueryPerformanceCounter((LARGE_INTEGER*)&h5);
	AVX(m5);
	QueryPerformanceCounter((LARGE_INTEGER*)&t5);
	cout << "AVX对齐: " << (t5 - h5) * 1000.0 / freq<< "ms" << endl;


	float** m6 = common();
	QueryPerformanceCounter((LARGE_INTEGER*)&h6);
	AVX_U(m6);
	QueryPerformanceCounter((LARGE_INTEGER*)&t6);
	cout << "AVX不对齐: " << (t6 - h6) * 1000.0 / freq<< "ms" << endl;


	return 0;
}