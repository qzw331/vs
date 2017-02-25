#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Public.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <time.h>
#include <sys/stat.h>
#include <map>
#include <random>
#include <cuda_occupancy.h>
#include "pthread.h"


#define SINGLE_GPU
typedef unsigned int    uint32;
typedef unsigned char   byte;
#define PER_FILE_SIZE 512 * 1024 * 1024
#define F(x, y, z)  ((z)^((x) &((y)^ (z))))
#define G(x, y, z)  ((y)^((z) &((x)^ (y))))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))
#define RL(x, y) (((x) << (y)) | ((x) >> (32 - (y))))  //x向左循环移y位
#define PP(x) (x<<24)|((x<<8)&0xff0000)|((x>>8)&0xff00)|(x>>24)  //将x高低位互换,例如PP(aabbccdd)=ddccbbaa
#define FF(a, b, c, d, x, s, ac) a = b + (RL((a + F(b,c,d) + x + ac),s))
#define GG(a, b, c, d, x, s, ac) a = b + (RL((a + G(b,c,d) + x + ac),s))
#define HH(a, b, c, d, x, s, ac) a = b + (RL((a + H(b,c,d) + x + ac),s))
#define II(a, b, c, d, x, s, ac) a = b + (RL((a + I(b,c,d) + x + ac),s))

__constant__  __int64  GPUColsNum, GPURowsNum;
__constant__  unsigned GPU_A = 0x67452301, GPU_B = 0xefcdab89, GPU_C = 0x98badcfe, GPU_D = 0x10325476; //初始化链接变量
__constant__  int table;
__constant__  unsigned __int64 m_nGPUPlainSpaceTotal;
__constant__  int nGPUPlainLenMin, nGPUPlainLenMax, GPUcharSetLen, GPURainbowTableIndex, GPUReduceOffset;
__constant__  unsigned __int64 m_nGPUPlainSpaceUpToX[MAX_PLAIN_LEN + 1];
__constant__  char  m_nGPUPlainCharset[128];
__constant__  char  GPUHashFunctionName[4], GPUCharSetName[50];



__device__ void IndexToPlain(unsigned __int64 m_nIndex, unsigned char *m_Plain, int *m_nPlainLen, int GPUPlainLenMax_Thread, 
	int GPUPlainLenMin_Thread, int GPUcharSetLen_Thread, unsigned __int64* GPUPlainSpaceUpToX_Thread)
{
	int i;
	for (i = GPUPlainLenMax_Thread - 1; i >= GPUPlainLenMin_Thread - 1; i--)
	{
		if (m_nIndex >= GPUPlainSpaceUpToX_Thread[i])//计算明文长度
		{
			*m_nPlainLen = i + 1;
			break;
		}
	}

	//若nIndexOfX已经小于32位能表示的最大值，则转换为32位求余计算
	uint64 nIndexOfX = m_nIndex - GPUPlainSpaceUpToX_Thread[*m_nPlainLen - 1];
	for (i = *m_nPlainLen - 1; i >= 0; i--)
	{
		if (nIndexOfX < 0x100000000ull)
			break;
		m_Plain[i] = m_nGPUPlainCharset[nIndexOfX % GPUcharSetLen_Thread];
		nIndexOfX /= GPUcharSetLen_Thread;
	}
	unsigned __int32 nIndex32 = (unsigned __int32)nIndexOfX;
	for (; i >= 0; i--)
	{
		m_Plain[i] = m_nGPUPlainCharset[nIndex32 % GPUcharSetLen_Thread];
		nIndex32 /= GPUcharSetLen_Thread;
	}
}

__device__ inline void MD5(unsigned char *pw, int m_nPlainLen, unsigned char *out, unsigned __int64* Index,
	int pos, unsigned __int64 GPUPlainSpaceTotal_Thread, int GPUReduceOffset_Thread)
{
	int i;
	unsigned int x[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	x[14] = m_nPlainLen << 3;

	unsigned char * toHashAsChar = (unsigned char*)x;
	for (i = 0; i<m_nPlainLen; i++) { toHashAsChar[i] = (unsigned char)*(pw + i); }
	toHashAsChar[m_nPlainLen] = 0x80;

	unsigned  a = 0x67452301, b = 0xefcdab89, c = 0x98badcfe, d = 0x10325476;
	/**//* Round 1 */
	FF(a, b, c, d, x[0], 7, 0xd76aa478); /**//* 1 */
	FF(d, a, b, c, x[1], 12, 0xe8c7b756); /**//* 2 */
	FF(c, d, a, b, x[2], 17, 0x242070db); /**//* 3 */
	FF(b, c, d, a, x[3], 22, 0xc1bdceee); /**//* 4 */
	FF(a, b, c, d, x[4], 7, 0xf57c0faf); /**//* 5 */
	FF(d, a, b, c, x[5], 12, 0x4787c62a); /**//* 6 */
	FF(c, d, a, b, x[6], 17, 0xa8304613); /**//* 7 */
	FF(b, c, d, a, x[7], 22, 0xfd469501); /**//* 8 */
	FF(a, b, c, d, x[8], 7, 0x698098d8); /**//* 9 */
	FF(d, a, b, c, x[9], 12, 0x8b44f7af); /**//* 10 */
	FF(c, d, a, b, x[10], 17, 0xffff5bb1); /**//* 11 */
	FF(b, c, d, a, x[11], 22, 0x895cd7be); /**//* 12 */
	FF(a, b, c, d, x[12], 7, 0x6b901122); /**//* 13 */
	FF(d, a, b, c, x[13], 12, 0xfd987193); /**//* 14 */
	FF(c, d, a, b, x[14], 17, 0xa679438e); /**//* 15 */
	FF(b, c, d, a, x[15], 22, 0x49b40821); /**//* 16 */
	/**//* Round 2 */
	GG(a, b, c, d, x[1], 5, 0xf61e2562); /**//* 17 */
	GG(d, a, b, c, x[6], 9, 0xc040b340); /**//* 18 */
	GG(c, d, a, b, x[11], 14, 0x265e5a51); /**//* 19 */
	GG(b, c, d, a, x[0], 20, 0xe9b6c7aa); /**//* 20 */
	GG(a, b, c, d, x[5], 5, 0xd62f105d); /**//* 21 */
	GG(d, a, b, c, x[10], 9, 0x02441453); /**//* 22 */
	GG(c, d, a, b, x[15], 14, 0xd8a1e681); /**//* 23 */
	GG(b, c, d, a, x[4], 20, 0xe7d3fbc8); /**//* 24 */
	GG(a, b, c, d, x[9], 5, 0x21e1cde6); /**//* 25 */
	GG(d, a, b, c, x[14], 9, 0xc33707d6); /**//* 26 */
	GG(c, d, a, b, x[3], 14, 0xf4d50d87); /**//* 27 */
	GG(b, c, d, a, x[8], 20, 0x455a14ed); /**//* 28 */
	GG(a, b, c, d, x[13], 5, 0xa9e3e905); /**//* 29 */
	GG(d, a, b, c, x[2], 9, 0xfcefa3f8); /**//* 30 */
	GG(c, d, a, b, x[7], 14, 0x676f02d9); /**//* 31 */
	GG(b, c, d, a, x[12], 20, 0x8d2a4c8a); /**//* 32 */
	/**//* Round 3 */
	HH(a, b, c, d, x[5], 4, 0xfffa3942); /**//* 33 */
	HH(d, a, b, c, x[8], 11, 0x8771f681); /**//* 34 */
	HH(c, d, a, b, x[11], 16, 0x6d9d6122); /**//* 35 */
	HH(b, c, d, a, x[14], 23, 0xfde5380c); /**//* 36 */
	HH(a, b, c, d, x[1], 4, 0xa4beea44); /**//* 37 */
	HH(d, a, b, c, x[4], 11, 0x4bdecfa9); /**//* 38 */
	HH(c, d, a, b, x[7], 16, 0xf6bb4b60); /**//* 39 */
	HH(b, c, d, a, x[10], 23, 0xbebfbc70); /**//* 40 */
	HH(a, b, c, d, x[13], 4, 0x289b7ec6); /**//* 41 */
	HH(d, a, b, c, x[0], 11, 0xeaa127fa); /**//* 42 */
	HH(c, d, a, b, x[3], 16, 0xd4ef3085); /**//* 43 */
	HH(b, c, d, a, x[6], 23, 0x04881d05); /**//* 44 */
	HH(a, b, c, d, x[9], 4, 0xd9d4d039); /**//* 45 */
	HH(d, a, b, c, x[12], 11, 0xe6db99e5); /**//* 46 */
	HH(c, d, a, b, x[15], 16, 0x1fa27cf8); /**//* 47 */
	HH(b, c, d, a, x[2], 23, 0xc4ac5665); /**//* 48 */
	/**//* Round 4 */
	II(a, b, c, d, x[0], 6, 0xf4292244); /**//* 49 */
	II(d, a, b, c, x[7], 10, 0x432aff97); /**//* 50 */
	II(c, d, a, b, x[14], 15, 0xab9423a7); /**//* 51 *////////////????????????????????
	II(b, c, d, a, x[5], 21, 0xfc93a039); /**//* 52 */
	II(a, b, c, d, x[12], 6, 0x655b59c3); /**//* 53 */
	II(d, a, b, c, x[3], 10, 0x8f0ccc92); /**//* 54 */
	II(c, d, a, b, x[10], 15, 0xffeff47d); /**//* 55 */
	II(b, c, d, a, x[1], 21, 0x85845dd1); /**//* 56 */
	II(a, b, c, d, x[8], 6, 0x6fa87e4f); /**//* 57 */
	II(d, a, b, c, x[15], 10, 0xfe2ce6e0); /**//* 58 */
	II(c, d, a, b, x[6], 15, 0xa3014314); /**//* 59 */
	II(b, c, d, a, x[13], 21, 0x4e0811a1); /**//* 60 */
	II(a, b, c, d, x[4], 6, 0xf7537e82); /**//* 61 */
	II(d, a, b, c, x[11], 10, 0xbd3af235); /**//* 62 */
	II(c, d, a, b, x[2], 15, 0x2ad7d2bb); /**//* 63 */
	II(b, c, d, a, x[9], 21, 0xeb86d391); /**//* 64 */

	a += 0x67452301, b += 0xefcdab89;//,c+=0x98badcfe,d+=0x10325476;

	unsigned __int64 nIndexOfX = b;
	*Index = ((nIndexOfX << 32) + a + GPUReduceOffset_Thread + pos) % GPUPlainSpaceTotal_Thread;
}

__global__ static void MD5RainBowKernel(RainbowChain* outBuffer)///参数可以修改一下！！
{
	__int64 GPUColsNum_Thread = GPUColsNum;
	__int64 GPURowsNum_Thread = GPURowsNum;
	int GPUPlainLenMax_Thread = nGPUPlainLenMax;
	int GPUPlainLenMin_Thread = nGPUPlainLenMin;
	int GPUcharSetLen_Thread = GPUcharSetLen;
	int GPURainbowTableIndex_Thread = GPURainbowTableIndex;
	int GPUReduceOffset_Thread = GPUReduceOffset;
	unsigned __int64 GPUPlainSpaceTotal_Thread = m_nGPUPlainSpaceTotal;
	unsigned __int64 GPUPlainSpaceUpToX_Thread[MAX_PLAIN_LEN + 1];
	for (int i = 0; i < MAX_PLAIN_LEN + 1; i++)
		GPUPlainSpaceUpToX_Thread[i] = m_nGPUPlainSpaceUpToX[i];

	unsigned __int64 num = 0;
	num = blockDim.x * blockIdx.x + threadIdx.x;
	if (num>GPURowsNum_Thread) return;

	unsigned char  m_Hash[16]; 
	unsigned char  m_Plain[MAX_PLAIN_LEN];
	int  m_nPlainLen;
	unsigned __int64  m_nIndex = (unsigned __int64)outBuffer[num].nIndexS;
	
	for (int nPos = 0; nPos<GPUColsNum_Thread - 1; nPos++) {
		IndexToPlain(m_nIndex, m_Plain, &m_nPlainLen, GPUPlainLenMax_Thread, GPUPlainLenMin_Thread,
			GPUcharSetLen_Thread, GPUPlainSpaceUpToX_Thread);
		//MD5PlainToHash(m_Plain, m_nPlainLen, m_Hash, &m_nIndex, nPos);
		MD5(m_Plain, m_nPlainLen, m_Hash, &m_nIndex, nPos, GPUPlainSpaceTotal_Thread, GPUReduceOffset_Thread);
	}
	outBuffer[num].nIndexE = m_nIndex;
}

//////////////////////////Host///////////////////////////////////////////
#define SUCCESS 1
#define FAIL 2
using namespace std;

unsigned char c2c(char c){ return (unsigned char)((c > '9') ? (c - 'a' + 10) : (c - '0')); }
bool SetPlainCharset(string sName, int nPlainLenMin, int nPlainLenMax, int *CharSetLen, unsigned __int64 * m_nPlainSpaceTotal, int GPU_num)
{
	int i;
	vector<string> vLine;
	char TemCharArray[128];
	__int64 m_nPlainSpaceUpToX[MAX_PLAIN_LEN + 1];
	/////////////////////////////////Get Charset Length//////////////////////////
	if (ReadLinesFromFile("charset.txt", vLine))
	{
		int i;
		for (i = 0; i < vLine.size(); i++)
		{
			// Filter comment
			if (vLine[i][0] == '#')
				continue;

			vector<string> vPart;
			if (SeperateString(vLine[i], "=", vPart))
			{
				// sCharsetName
				string sCharsetName = TrimString(vPart[0]);
				if (sCharsetName == "")
					continue;

				// sCharsetName charset check
				bool fCharsetNameCheckPass = true;
				int j;
				for (j = 0; j < sCharsetName.size(); j++)
				{
					if (!isalpha(sCharsetName[j])
						&& !isdigit(sCharsetName[j])
						&& (sCharsetName[j] != '-'))
					{
						fCharsetNameCheckPass = false;
						break;
					}
				}
				if (!fCharsetNameCheckPass)
				{
					printf("invalid charset name %s in charset configuration file\n", sCharsetName.c_str());
					continue;
				}

				// sCharsetContent
				string sCharsetContent = TrimString(vPart[1]);
				if (sCharsetContent == "" || sCharsetContent == "[]")
					continue;
				if (sCharsetContent[0] != '[' || sCharsetContent[sCharsetContent.size() - 1] != ']')
				{
					printf("invalid charset content %s in charset configuration file\n", sCharsetContent.c_str());
					continue;
				}
				sCharsetContent = sCharsetContent.substr(1, sCharsetContent.size() - 2);
				if (sCharsetContent.size() > 256)
				{
					printf("charset content %s too long\n", sCharsetContent.c_str());
					continue;
				}

				//printf("%s = [%s]\n", sCharsetName.c_str(), sCharsetContent.c_str());

				// Is it the wanted charset?
				if (sCharsetName == sName)
				{
					*CharSetLen = sCharsetContent.size();
					memcpy(TemCharArray, sCharsetContent.c_str(), *CharSetLen);
				}
			}
		}
	}
	else
		printf("can't open charset configuration file\n");

	if (nPlainLenMin < 1 || nPlainLenMax > MAX_PLAIN_LEN || nPlainLenMin > nPlainLenMax)
	{
		printf("invalid plaintext length range: %d - %d\n", nPlainLenMin, nPlainLenMax);
		return false;
	}

	m_nPlainSpaceUpToX[0] = 0;
	uint64 nTemp = 1;

	for (i = 1; i <= nPlainLenMax; i++)
	{
		nTemp *= (*CharSetLen);
		if (i < nPlainLenMin) 	m_nPlainSpaceUpToX[i] = 0;
		else  m_nPlainSpaceUpToX[i] = m_nPlainSpaceUpToX[i - 1] + nTemp;
	}
	*m_nPlainSpaceTotal = m_nPlainSpaceUpToX[nPlainLenMax];
	int aa = *CharSetLen;
	for (int i = 0; i<GPU_num - 1; i++)//新增
	{
		cudaSetDevice(i);
		cudaMemcpyToSymbol(GPUcharSetLen, &aa, sizeof(int), 0, cudaMemcpyHostToDevice);
		unsigned __int64 ab = *m_nPlainSpaceTotal;
		cudaMemcpyToSymbol(m_nGPUPlainSpaceTotal, &ab, sizeof(unsigned __int64), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(m_nGPUPlainSpaceUpToX, m_nPlainSpaceUpToX, sizeof(unsigned __int64)* (MAX_PLAIN_LEN + 1), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(m_nGPUPlainCharset, TemCharArray, *CharSetLen* sizeof(char), 0, cudaMemcpyHostToDevice);
	}
	return true;
}

void GetGPUInfo(cudaDeviceProp &prop, int GPU_N)
{
	for (int igpu = 0; igpu < GPU_N - 1; igpu++)
	{
		cudaGetDeviceProperties(&prop, igpu);
		cout << "GPU name: " << prop.name << endl
			<< "total globle mem: " << prop.totalGlobalMem / 1024 / 1024 << " MB" << endl
			<< "SM count: " << prop.multiProcessorCount << endl
			<< "warp size: " << prop.warpSize << endl
			<< "max thread per block: " << prop.maxThreadsPerBlock << endl
			<< "max thread per SM: " << prop.maxThreadsPerMultiProcessor << endl
			<< "max grid size: " << prop.maxGridSize[0] << "*" << prop.maxGridSize[1] << "*" << prop.maxGridSize[2] << endl
			<< "shared mem per SM: " << prop.sharedMemPerMultiprocessor << " Bytes" << endl
			<< "shared mem per block: " << prop.sharedMemPerBlock << " Bytes" << endl
			<< "device overlap" << prop.deviceOverlap << endl;
	}
	
}

TGPUplan plan[MAX_GPU_COUNT];
FILE *logfp;
char filename[256];
FILE   *tablefile = NULL;
int FileNumber = 0;
int currentFileSize = 0;
int GPU_N = 0;
unsigned __int64 RainbowChainOfGPU;
int nNumberTable;
unsigned __int64 AllChainNumber;
unsigned __int64 RainbowChainOfEachTable, RainbowChainLength;
int m_nPlainCharsetLen;
unsigned __int64 m_nPlainSpaceTotal;
unsigned __int64 KeySpceSizeN;
unsigned char PlainCharset[256];
unsigned char PlainCharsetName[20];

char* sHashRoutineName = "md5";
char* sCharsetName = "all85";
int nPlainLenMin = 1;
int nPlainLenMax = 4;
double AllTableSize = 0.00001;
double CoverProbality = 0.96;
int nRainbowTableIndex = 1;
char* sFileTitleSuffix = "test";

struct HtoDPARA{
	unsigned int curBuffer;
	std::uniform_int_distribution<unsigned __int64> dis;
	std::mt19937 gen;
	HtoDPARA(unsigned int i,
		std::uniform_int_distribution<unsigned __int64> d,
		std::mt19937 g){
		curBuffer = i;
		dis = d;
		gen = g;
	}
};

pthread_t HtoD_Thread;
pthread_t DtoH_Thread;
pthread_t GenFile_Thread;
pthread_t Launch_Thread;

int  BlockSize =  512;
int  GridSize = 512;

int HtoD(unsigned int curBuffer, std::uniform_int_distribution<unsigned __int64> &dis, std::mt19937 &gen)
{
	cudaError cudaStatus;
	cout << "HtoD " << curBuffer << " ";
	for (int igpu = 0; igpu < GPU_N - 1; igpu++)  {
		for (int j = 0; j < RainbowChainOfGPU; j++) {
			plan[igpu].h_Data[curBuffer][j].nIndexS = (unsigned __int64)dis(gen);
			plan[igpu].h_Data[curBuffer][j].nIndexE = 0;
		}
	}
	for (int igpu = 0; igpu < GPU_N - 1; igpu++)  {
		cudaSetDevice(igpu);
		cudaStatus = cudaMemcpyAsync(plan[igpu].d_Data[curBuffer], plan[igpu].h_Data[curBuffer], RainbowChainOfGPU * sizeof(RainbowChain), cudaMemcpyHostToDevice, plan[igpu].stream[curBuffer]);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "host to device %d cudaMemcpyAsync  failed!", igpu);
			fprintf(logfp, "host to device %d cudaMemcpyAsync  failed!", igpu);
			return FAIL;
		}
		cudaStreamSynchronize(plan[igpu].stream[curBuffer]);
	}
	cout << "finish " << endl;
	return SUCCESS;
}

int LaunchKernel(unsigned int curBuffer)
{
	cout << "kernel " << curBuffer << " ";
	for (int igpu = 0; igpu < GPU_N - 1; igpu++)  {
		cudaSetDevice(igpu);
		cudaStreamSynchronize(plan[igpu].stream[curBuffer]);
		MD5RainBowKernel <<<GridSize, BlockSize, 0, plan[igpu].stream[curBuffer] >>>(plan[igpu].d_Data[curBuffer]);
	}
	cout << "finish " << endl;
	return SUCCESS;
}

int DtoH(unsigned int curBuffer)
{
	cout << "DtoH " << curBuffer << " ";
	cudaError cudaStatus;
	for (int igpu = 0; igpu < GPU_N - 1; igpu++)  {
		cudaSetDevice(igpu);
		cudaStatus = cudaMemcpyAsync(plan[igpu].h_Data[curBuffer], plan[igpu].d_Data[curBuffer], RainbowChainOfGPU * sizeof(RainbowChain), cudaMemcpyDeviceToHost, plan[igpu].stream[curBuffer]);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Device Copy device to host %d cudaMemcpyAsync failed!", igpu);
			fprintf(logfp, "Device Copy device to host %d cudaMemcpyAsync failed!", igpu);
			return FAIL;
		}
		cudaStreamSynchronize(plan[igpu].stream[curBuffer]);
	}
	cout << "finish" << endl;
	return SUCCESS;
}

int GenFile(unsigned int curBuffer)
{
	cout << "Gen File " << curBuffer << " ";
	for (int igpu = 0; igpu < GPU_N - 1; igpu++)
	{
		cudaStreamSynchronize(plan[igpu].stream[curBuffer]);
		cudaSetDevice(igpu);
		if (currentFileSize == 0)
		{
			sprintf(filename, "%s_%s_%I64d_%I64d_%d_%d_%d_%d_%s.rt",
				sHashRoutineName, sCharsetName,
				RainbowChainOfEachTable, RainbowChainLength, nRainbowTableIndex, nPlainLenMin, nPlainLenMax, FileNumber++, sFileTitleSuffix);

			printf("filename=%s is generating \n", filename);
			fprintf(logfp, "filename=%s is generating \n", filename);

			fclose(fopen(filename, "a"));
			if (!(tablefile = fopen(filename, "r+b")))  { perror(filename); return FAIL; }
		}

		for (int j = 0; j < RainbowChainOfGPU; j++)
		{
			fwrite(&plan[igpu].h_Data[curBuffer][j].nIndexS, 1, 8, tablefile);
			fwrite(&plan[igpu].h_Data[curBuffer][j].nIndexE, 1, 8, tablefile);
		}

		currentFileSize = currentFileSize + RainbowChainOfGPU*sizeof(RainbowChain);
		if (currentFileSize>PER_FILE_SIZE){
			fclose(tablefile);
			currentFileSize = 0;
			cout << endl << filename << endl;
		}
	}
	fflush(tablefile);
	cout << "Gen file finish return " << endl;
	return SUCCESS;
}

void* HtoD_Thread_Fun(void *para)
{
	int *retVal = new int;
	pthread_join(GenFile_Thread, (void**)&retVal);
	if (*(int*)retVal == FAIL)
		return (void*)retVal;
	*retVal = HtoD(((struct HtoDPARA*)para)->curBuffer, ((struct HtoDPARA*)para)->dis, ((struct HtoDPARA*)para)->gen);
	if (*(int*)retVal == FAIL)
		return (void*)retVal;
	*retVal = SUCCESS;
	//return (void*)retVal;
	pthread_exit((void*)retVal);
}

void* DtoH_Thread_Fun(void *curBuffer)
{
	int *retVal = new int;
	*retVal = DtoH(*(unsigned int*)curBuffer);
	if (*(int*)retVal == FAIL)
		return (void*)retVal;
	*retVal = SUCCESS;
	//return (void*)retVal;
	pthread_exit((void*)retVal);
}

void* GenFile_Thread_Fun(void *curBuffer)
{
	int *retVal = new int;
	*retVal = GenFile(*(unsigned int *)curBuffer);
	if (*(int*)retVal == FAIL)
		return (void*)retVal;
	*retVal = SUCCESS;
	//return (void*)retVal;
	pthread_exit((void*)retVal);
}

void* Launch_Thread_Fun(void *curBuffer)
{
	int *retVal = new int;
	*retVal = LaunchKernel(*(unsigned int *)curBuffer);
	if (*(int*)retVal == FAIL)
		return (void*)retVal;
	*retVal = SUCCESS;
	//return (void*)retVal;
	pthread_exit((void*)retVal);
}

int main(int argc, char **argv)
{
	cudaError_t cudaStatus;
	time_t nowtime;    struct tm* ptm;
	cudaDeviceProp prop;
	//if (argc != 9)
	//{
	//	Usage();
	//	system("pause");
	//	return -1;
	//}
/*
	sHashRoutineName = argv[1];
	sCharsetName = argv[2];
	nPlainLenMin = atoi(argv[3]);
	nPlainLenMax = atoi(argv[4]);
	AllTableSize = atof(argv[5]);
	CoverProbality = atof(argv[6]);
	nRainbowTableIndex = atoi(argv[7]);
	sFileTitleSuffix = argv[8];
*/
	logfp = fopen(".\\RaionBowTableGenlog.txt", "a");

	cudaDeviceReset();
	cudaGetDeviceCount(&GPU_N);
	if (GPU_N > MAX_GPU_COUNT)  { GPU_N = MAX_GPU_COUNT; }
	printf("CUDA-capable device count: %d\n", GPU_N);
	fprintf(logfp, "CUDA-capable device count: %d\n", GPU_N);

#ifdef SINGLE_GPU
	GPU_N = 2;//single GPU test
#endif

	GetGPUInfo(prop,GPU_N);

	SetPlainCharset(sCharsetName, nPlainLenMin, nPlainLenMax, &m_nPlainCharsetLen,
		&m_nPlainSpaceTotal, GPU_N);

	Calculator(AllTableSize, CoverProbality,
		nNumberTable, RainbowChainOfEachTable, AllChainNumber, RainbowChainLength, nPlainLenMin, nPlainLenMax,
		m_nPlainCharsetLen, m_nPlainSpaceTotal, logfp);


	if (nRainbowTableIndex>nNumberTable)
	{
		printf("The Index of Table Beyond the Table Number, Re-input Again!\n");
		fprintf(logfp, "The Index of Table Beyond the Table Number, Re-input Again!\n");
		return -1;
	}

	//计算每张表在生成过程中循环次数
	RainbowChainOfGPU = BlockSize*GridSize;
	int MainLoopNumber = (RainbowChainOfEachTable + RainbowChainOfGPU*(GPU_N - 1) - 1) / RainbowChainOfGPU / (GPU_N - 1);
	unsigned __int64 Remainer = RainbowChainOfEachTable % (RainbowChainOfGPU*(GPU_N - 1));

	for (int i = 0; i<GPU_N - 1; i++)
	{
		cudaSetDevice(i);
		cudaMemcpyToSymbol(GPURowsNum, &RainbowChainOfGPU, sizeof(__int64), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(GPUColsNum, &RainbowChainLength, sizeof(__int64), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(GPUHashFunctionName, sHashRoutineName, sizeof(char)* strlen(sHashRoutineName), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(GPUCharSetName, sCharsetName, sizeof(char)*  strlen(sCharsetName), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(nGPUPlainLenMin, &nPlainLenMin, sizeof(int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(nGPUPlainLenMax, &nPlainLenMax, sizeof(int), 0, cudaMemcpyHostToDevice);
	}

	printf("Gridesize: %d   Blocksize :%d    ", GridSize, BlockSize);
	fprintf(logfp, "Gridesize: %d   Blocksize :%d    ", GridSize, BlockSize);

	cout << "BlockSize " << BlockSize << endl;
	cout << "GridSize " << GridSize << endl;
	cout << "RainbowChainOfGPU " << RainbowChainOfGPU << endl;


	time(&nowtime);
	ptm = localtime(&nowtime);
	printf("———————————————————————————————\n");
	fprintf(logfp, "———————————————————————————————\n");
	printf("计算开始时间：%.2d年 %.2d月  %.2d日  %.2d 时: %.2d分  %.2d秒\n",
		ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
	fprintf(logfp, "计算开始时间：%.2d年 %.2d月  %.2d日  %.2d 时: %.2d分  %.2d秒\n",
		ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
	printf("———————————————————————————————\n");
	fprintf(logfp, "———————————————————————————————\n");

	for (int igpu = 0; igpu < GPU_N - 1; igpu++)
	{
		cudaSetDevice(igpu);
		cudaStreamCreate(&plan[igpu].stream[0]);
		cudaStreamCreate(&plan[igpu].stream[1]);
		cudaStatus = cudaHostAlloc((void**)&plan[igpu].h_Data[0], RainbowChainOfGPU *sizeof(RainbowChain), cudaHostAllocDefault);
		cudaStatus = cudaHostAlloc((void**)&plan[igpu].h_Data[1], RainbowChainOfGPU *sizeof(RainbowChain), cudaHostAllocDefault);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, " Malloc host memroy for devive %d is failed!", igpu);
			fprintf(logfp, " Malloc host memroy for devive %d is failed!", igpu);
			goto Error;
		}

		cudaStatus = cudaMalloc((void **)&plan[igpu].d_Data[0], RainbowChainOfGPU * sizeof(RainbowChain));
		cudaStatus = cudaMalloc((void **)&plan[igpu].d_Data[1], RainbowChainOfGPU * sizeof(RainbowChain));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Device %d cudaMem  malloc is failed!", igpu);
			fprintf(logfp, "Device %d cudaMem  malloc is failed!", igpu);
			goto Error;
		}
	}
	//////////////////////初始化随机数生成器///////////////////////////	

	unsigned __int64 LowerBounce = 1, UpperBounce = 1;
	for (int i = 1; i < nPlainLenMin; i++){ LowerBounce *= m_nPlainCharsetLen; }
	for (int i = 1; i <= nPlainLenMax; i++){ UpperBounce *= m_nPlainCharsetLen; }

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<unsigned __int64> dis(LowerBounce, UpperBounce);

	for (; nRainbowTableIndex <= nNumberTable; nRainbowTableIndex++)
	{
		printf("———————————————————————————————\n");
		fprintf(logfp, "———————————————————————————————\n");
		printf("|               正在计算第%d张表，一共%d张表                      |\n", nRainbowTableIndex, nNumberTable);
		fprintf(logfp, "|               正在计算第%d张表，一共%d张表                     |\n", nRainbowTableIndex, nNumberTable);
		printf("———————————————————————————————*\n\n\n");
		fprintf(logfp, "———————————————————————————————*\n\n\n");

		for (int igpu = 0; igpu < GPU_N - 1; igpu++)//彩虹表生成的Reduce过程中，即HashToIndex用到GPUReduceOffset参数
		{
			cudaSetDevice(igpu);
			cudaMemcpyToSymbol(GPURainbowTableIndex, &nRainbowTableIndex, sizeof(nRainbowTableIndex), 0, cudaMemcpyHostToDevice);
			int temp = (nRainbowTableIndex)* 65536;
			cudaMemcpyToSymbol(GPUReduceOffset, &temp, sizeof(temp), 0, cudaMemcpyHostToDevice);
			RainbowChainOfGPU = BlockSize*GridSize;
			cudaMemcpyToSymbol(GPURowsNum, &RainbowChainOfGPU, sizeof(__int64), 0, cudaMemcpyHostToDevice);
		}

		//double buffer
		//for (iloop = 0; iloop <= MainLoopNum; iloop++)
		//{
		//	if (iloop == 0)		HtoD[cur];
		//	if (iloop != 0)		DtoH[~cur];
		//	if (iloop != last)	kernel[cur];
		//	if (iloop != 0)		GenFile[~cur];
		//	if (iloop != last && iloop != last - 1)	HtoD[~cur];
		//	wait[cur];
		//	cur ^= 1;
		//}
		cudaDeviceSynchronize();
		bool flag = false;
		unsigned int curBuffer = 0;
		unsigned int lastBuffer = 1;
		HtoDPARA para(curBuffer, dis, gen);
		int* threadRes;

		//para.curBuffer = curBuffer;
		//pthread_create(&HtoD_Thread, NULL, HtoD_Thread_Fun, &para);
		//pthread_join(HtoD_Thread, NULL);
		//cudaStreamSynchronize(plan[1].stream[curBuffer]);
		//LaunchKernel(curBuffer);
		//para.curBuffer = lastBuffer;
		//pthread_create(&HtoD_Thread, NULL, HtoD_Thread_Fun, &para);
		//cudaStreamSynchronize(plan[1].stream[curBuffer]);
		//pthread_join(HtoD_Thread, NULL);
		////loop 0 finish
		//pthread_create(&DtoH_Thread, NULL, DtoH_Thread_Fun, &curBuffer);
		//LaunchKernel(lastBuffer);
		//para.curBuffer = curBuffer;
		//pthread_create(&HtoD_Thread, NULL, HtoD_Thread_Fun, &para);
		//cudaStreamSynchronize(plan[1].stream[lastBuffer]);
		//pthread_join(DtoH_Thread, NULL);
		//pthread_join(GenFile_Thread, NULL);
		//pthread_join(HtoD_Thread, NULL);
		////loop 2 finish
		//pthread_create(&DtoH_Thread, NULL, DtoH_Thread_Fun, &lastBuffer);
		//LaunchKernel(curBuffer);
		//pthread_create(&GenFile_Thread, NULL, GenFile_Thread_Fun, &lastBuffer);
		//para.curBuffer = lastBuffer;
		//pthread_create(&HtoD_Thread, NULL, HtoD_Thread_Fun, &para);
		//cudaStreamSynchronize(plan[1].stream[curBuffer]);
		//pthread_join(DtoH_Thread, NULL);
		//pthread_join(GenFile_Thread, NULL);
		//pthread_join(HtoD_Thread, NULL);
		////loop 3 finish 
		
		for (int iloop = 0; iloop <= MainLoopNumber; iloop++)//<=MainLoopNum,因为最后一次不调用kernel
		{
			time(&nowtime);
			ptm = localtime(&nowtime);
			printf("———————————————————————————————\n");
			fprintf(logfp, "———————————————————————————————\n");
			printf("|               正在计算第%d张表的第%d轮，一共%d张表，每张%d轮\n\t\t\t%.2d年 %.2d月 %.2d日 %.2d时 %.2d分 %.2d秒                      |\n", nRainbowTableIndex, iloop + 1, nNumberTable, MainLoopNumber,
				ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
			fprintf(logfp, "|               正在计算第%d张表的第%d轮，一共%d张表，每张%d轮\n\t\t\t%.2d年 %.2d月 %.2d日 %.2d时 %.2d分 %.2d秒                     |\n", nRainbowTableIndex, iloop + 1, nNumberTable, MainLoopNumber,
				ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
			printf("———————————————————————————————*\n\n\n");
			fprintf(logfp, "———————————————————————————————*\n\n\n");
			if (flag)
			{
				curBuffer = 1;
				lastBuffer = 0;
			}
			else
			{
				curBuffer = 0;
				lastBuffer = 1;
			}

			if (iloop == MainLoopNumber - 1 && Remainer>0){
				RainbowChainOfGPU = ((unsigned __int64)(Remainer / (GPU_N - 1)));
				for (int igpu = 1; igpu < GPU_N; igpu++)
				{
					cudaSetDevice(igpu);
					cudaMemcpyToSymbol(GPURowsNum, &RainbowChainOfGPU, sizeof(__int64), 0, cudaMemcpyHostToDevice);
				}
			}

			if (iloop == 0)//HtoD[cur]
			{
				para.curBuffer = curBuffer;
				pthread_create(&HtoD_Thread, NULL, HtoD_Thread_Fun, &para);
				pthread_join(HtoD_Thread, (void**)&threadRes);
				if (*threadRes == FAIL)
					goto Error;
				for (int igpu = 1; igpu < GPU_N; igpu++)
					cudaStreamSynchronize(plan[igpu].stream[curBuffer]);
			}

			if (iloop != 0)//DtoH[~cur]
			{
				pthread_create(&DtoH_Thread, NULL, DtoH_Thread_Fun, &lastBuffer);
			}

			if (iloop != MainLoopNumber)//kernel[cur]
			{
				LaunchKernel(curBuffer);
			}

			if (iloop != 0)//GenFile[~cur]
			{
				pthread_create(&GenFile_Thread, NULL, GenFile_Thread_Fun, &lastBuffer);
			}
			if (iloop != MainLoopNumber && iloop != MainLoopNumber - 1)//HtoD[~cur]
			{
				para.curBuffer = lastBuffer;
				pthread_create(&HtoD_Thread, NULL, HtoD_Thread_Fun, &para);
			}

			cout << "等待kernel计算完成" << endl;

			for (int igpu = 0; igpu < GPU_N - 1; igpu++)
			{
				cudaStreamSynchronize(plan[igpu].stream[curBuffer]);
			}
			pthread_join(DtoH_Thread, (void**)threadRes);
			if (*threadRes == FAIL)
				goto Error;
			cout << "DtoH thread join finish"<<endl;
			//pthread_join(GenFile_Thread, (void**)threadRes);
			//if (*threadRes == FAIL)
			//	goto Error;
			if (iloop != MainLoopNumber && iloop != MainLoopNumber - 1)
			{
				pthread_join(HtoD_Thread, (void**)threadRes);
				if (*threadRes == FAIL)
					goto Error;
				cout << "HtoD thread join finish" << endl;
			}
			if (iloop == MainLoopNumber || iloop == MainLoopNumber - 1)
			{
				pthread_join(GenFile_Thread, (void**)threadRes);
				if (*threadRes == FAIL)
					goto Error;
				cout << "Gen file thread join finish " <<endl;
			}
			cout << endl << endl;
			flag ^= 1;

			time(&nowtime);
			ptm = localtime(&nowtime);
			printf("——————————————————————————————\n");
			fprintf(logfp, "——————————————————————————————\n");
			printf("|   第%d张表第%d轮计算完成，结束时间 %.2d年 %.2d月 %.2d日 %.2d时 %.2d分 %.2d秒   |\n", nRainbowTableIndex, iloop + 1,
				ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
			fprintf(logfp, "|   第%d张表第%d轮计算完成，结束时间 %.2d年 %.2d月 %.2d日 %.2d时 %.2d分 %.2d秒   |\n", nRainbowTableIndex, iloop + 1,
				ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
			printf("——————————————————————————————\n\n\n");
			fprintf(logfp, "——————————————————————————————\n\n\n");
			
			//if (iloop == 3)
			//	break;
		}
		//某一张表生成完成，关闭文件
		fclose(tablefile);
		currentFileSize = 0;
		//break;
	}

	////////////////////////////////////////////////////////////////////////////	
	time(&nowtime);
	ptm = localtime(&nowtime);
	printf("——————————————————————————————\n");
	fprintf(logfp, "——————————————————————————————\n");
	printf("|   rows number: %I64d cols: %I64d \n 结束时间 %.2d年 %.2d月 %.2d日 %.2d时 %.2d分 %.2d秒   |\n", RainbowChainOfEachTable, RainbowChainLength,
		ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
	fprintf(logfp, "|   rows number: %I64d cols: %I64d \n 结束时间 %.2d年 %.2d月 %.2d日 %.2d时 %.2d分 %.2d秒   |\n", RainbowChainOfEachTable, RainbowChainLength,
		ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
	printf("——————————————————————————————\n\n\n");
	fprintf(logfp, "——————————————————————————————\n\n\n");
	////////////////////////////////////////////////////////////////////////////

	                                                                                                            
Error:
	if (tablefile != NULL)  fclose(tablefile);
	for (int igpu = 0; igpu < GPU_N - 1; igpu++)
	{
		cudaSetDevice(igpu);
		cudaStreamDestroy(plan[igpu].stream[0]);
		cudaStreamDestroy(plan[igpu].stream[1]);
		cout <<"stream destory finish"<<endl;       
		cudaFreeHost(plan[igpu].h_Data[0]);
		cudaFreeHost(plan[igpu].h_Data[1]);
		cout <<"free h_data finish"<<endl;
		cudaFree(plan[igpu].d_Data[0]);
		cudaFree(plan[igpu].d_Data[1]);
		cout <<"free d_data finish" <<endl;                                                                               
		cudaDeviceReset();
		cout << "device reset finish" <<endl;
	}

	printf("|——————————Over! Thanks!————————————————————|\n\n\n");
	fprintf(logfp, "|——————————Over! Thanks!————————————————————|\n\n\n");

	fclose(logfp);
	return 0;
}
