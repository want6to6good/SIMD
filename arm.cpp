#include<iostream>
#include<sys/time.h>
#include<arm_neon.h>  // neon

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
	float** m = (float**)aligned_alloc(32 * n * sizeof(float**), 32);
	for (int i = 0; i < n; i++)
		m[i] = (float*)aligned_alloc(32 * n * sizeof(float*), 32);
	init(m);
	return m;
}

void gauss(float** a)  // 标准的高斯消去算法
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

void neon(float** a) // 对齐
{  
	float32x4_t va, vt, vaik, vakj, vaij, vx;
	for (int k = 0; k < n; k++)
	{
		vt = vmovq_n_f32(a[k][k]);  
		int j = k + 1;
		for (; j + 4 < n; j += 4)
		{
			va = vld1q_f32(&a[k][j]);
			va = vdivq_f32(va, vt);
			vst1q_f32((float32_t*)&a[k][j], va);
		}
		for (; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];  
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = vmovq_n_f32(a[i][k]);
			for (j = k + 1; j + 4 < n; j += 4)
			{
				vakj = vld1q_f32(&a[k][j]);
				vaij = vld1q_f32(&a[i][j]);
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32((float32_t*)&a[i][j], vaij);
			}
			for (; j < n; j++)
				a[i][j] -= a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}
}

void neon_U(float** a)// 未对齐
{  
	float32x4_t va, vt, vaik, vakj, vaij, vx;
	for (int k = 0; k < n; k++)
	{
		vt = vmovq_n_f32(a[k][k]);  
		int j = 0;
		for (j = k + 1; j + 4 < n; j += 4)
		{
			va = vld1q_f32(&a[k][j]);
			va = vdivq_f32(va, vt);
			vst1q_f32((float32_t*)&a[k][j], va);
		}
		for (j; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];  // 善后
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = vmovq_n_f32(a[i][k]);
			for (j = k + 1; j + 4 < n; j += 4)
			{
				vakj = vld1q_f32(&a[k][j]);
				vaij = vld1q_f32(&a[i][j]);
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32((float32_t*)&a[i][j], vaij);
			}
			for (; j < n; j++)
				a[i][j] -= a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}
}

void neon_fms(float** a) // 使用fmsq 
{ 
	float32x4_t va, vt, vaik, vakj, vaij;
	for (int k = 0; k < n; k++)
	{
		vt = vmovq_n_f32(a[k][k]); 
		int j = k + 1;
		for (; j + 4 < n; j += 4)
		{
			va = vld1q_f32(&a[k][j]);
			va = vdivq_f32(va, vt);
			vst1q_f32((float32_t*)&a[k][j], va);
		}
		for (; j < n; j++)
			a[k][j] = a[k][j] / a[k][k]; 
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = vmovq_n_f32(a[i][k]);
			for (j = k + 1; j + 4 < n; j += 4)
			{
				vakj = vld1q_f32(&a[k][j]);
				vaij = vld1q_f32(&a[i][j]);
				vaij = vfmsq_f32(vaij, vakj, vaik);
				vst1q_f32((float32_t*)&a[i][j], vaij);
			}
			for (; j < n; j++)
				a[i][j] -= a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}
}

void neon_Opt(float** a)// 尝试采用流水线优化
{  
	float32x4_t va1, va2, vt1, vt2, vaik1, vaik2, vakj1, vakj2, vaij1, vaij2, vx1, vx2;
	for (int k = 0; k < n; k++)
	{
		vt1 = vmovq_n_f32(a[k][k]);  
		vt2 = vmovq_n_f32(a[k][k]); 
		int j =  k + 1;
		for (; j + 8 < n; j += 8)
		{
			va1 = vld1q_f32(&a[k][j]);
			va2 = vld1q_f32(&a[k][j] + 4);
			va1 = vdivq_f32(va1, vt1);
			va2 = vdivq_f32(va2, vt2);
			vst1q_f32((float32_t*)&a[k][j], va1);
			vst1q_f32((float32_t*)(&a[k][j] + 4), va2); 
		}
		for (; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik1 = vmovq_n_f32(a[i][k]);
			vaik2 = vmovq_n_f32(a[i][k]);
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj1 = vld1q_f32(&a[k][j]);
				vakj2 = vld1q_f32(&a[k][j] + 4);
				vaij1 = vld1q_f32(&a[i][j]);
				vaij2 = vld1q_f32(&a[i][j] + 4);
				vx1 = vmulq_f32(vakj1, vaik1);
				vx2 = vmulq_f32(vakj2, vaik2);
				vaij1 = vsubq_f32(vaij1, vx1);
				vaij2 = vsubq_f32(vaij2, vx2);
				vst1q_f32((float32_t*)&a[i][j], vaij1);
				vst1q_f32((float32_t*)(&a[i][j] + 4), vaij2); 
			}
			for (; j < n; j++)
				a[i][j] -= a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}
}


int main() 
{
	cin >> n;
	cout << "问题规模为" << n << "*" <<n<<"的矩阵" <<endl;

	init_();

	float time_use = 0;
	struct timeval start1, start2, start3, start4, start5, start6, start7, start8;
	struct timeval end1, end2, end3, end4, end5, end6, end7, end8;
	

	cout << endl << endl;


	float** m1 = common();
	gettimeofday(&start1, NULL); 
	gauss(m1);
	gettimeofday(&end1, NULL); 
	time_use = (end1.tv_sec - start1.tv_sec) * 1000000 + (end1.tv_usec - start1.tv_usec);
	cout << "GE: " << time_use / 1000<< "ms" << endl;


	float** m2 = common();
	gettimeofday(&start2, NULL); 
	gauss_cache(m2);
	gettimeofday(&end2, NULL); 
	time_use = (end2.tv_sec - start2.tv_sec) * 1000000 + (end2.tv_usec - start2.tv_usec);
	cout << "C_GE: " << time_use / 1000<< "ms" << endl;
	

	float** m3 = special();
	gettimeofday(&start3, NULL); 
	neon(m3);
	gettimeofday(&end3, NULL); 
	time_use = (end3.tv_sec - start3.tv_sec) * 1000000 + (end3.tv_usec - start3.tv_usec);
	cout << "neon_GE: " << time_use / 1000<< "ms" << endl;


	float** m4 = common();
	gettimeofday(&start4, NULL); 
	neon_U(m4);
	gettimeofday(&end4, NULL); 
	time_use = (end4.tv_sec - start4.tv_sec) * 1000000 + (end4.tv_usec - start4.tv_usec);
	cout << "neon_U_GE: " << time_use / 1000<< "ms" << endl;


	float** m8 = common();
	gettimeofday(&start8, NULL); 
	neon_fms(m8);
	gettimeofday(&end8, NULL); 
	time_use = (end8.tv_sec - start8.tv_sec) * 1000000 + (end8.tv_usec - start8.tv_usec);
	cout << "neon_U_GE_fms: " << time_use / 1000 << "ms" << endl;


	float** m7 = common();
	gettimeofday(&start7, NULL); 
	neon_Opt(m7);
	gettimeofday(&end7, NULL); 
	time_use = (end7.tv_sec - start7.tv_sec) * 1000000 + (end7.tv_usec - start7.tv_usec);/
	cout << "neon_U_GE_Opt1: " << time_use / 1000<< "ms" << endl;


	return 0;
}