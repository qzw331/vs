/*
   RainbowCrack - a general propose implementation of Philippe Oechslin's faster time-memory trade-off technique.

   Copyright (C) Zhu Shuanglei <shuanglei@hotmail.com>
*/

#ifndef _PUBLIC_H
#define _PUBLIC_H

#include <stdio.h>
#include "cuda_runtime.h"
#include <string>
#include <vector>
#include <list>
#include <pthread.h>
using namespace std;

#ifdef _WIN32
	#define uint64 unsigned __int64
#else
	#define uint64 u_int64_t
#endif

//int
#define MAX_PLAIN_LEN 13
#define MIN_HASH_LEN  8
#define MAX_HASH_LEN  256

const int MAX_GPU_COUNT = 32;

#define oneTera                   1024*1024*1024*1024


struct RainbowChain
{
	uint64 nIndexS;
	uint64 nIndexE;
};

typedef struct {
	RainbowChain  *h_Data[2], *d_Data[2];   
    cudaStream_t stream[2];
	cudaEvent_t DtoHDone[2];
} TGPUplan;

unsigned int GetFileLen(FILE* file);
string TrimString(string s);
bool ReadLinesFromFile(string sPathName, vector<string>& vLine);
bool SeperateString(string s, string sSeperator, vector<string>& vPart);
string uint64tostr(uint64 n);
string uint64tohexstr(uint64 n);
string HexToStr(const unsigned char* pData, int nLen);

bool Calculator(double AllTableSize, double CoverProbality, int &nNumberTable, unsigned __int64 &RainbowChainOfEachTable,
	unsigned __int64 &AllChainNumber, unsigned __int64 &RainbowChainLength, int nPlainLenMin, int nPlainLenMax,
	int m_nPlainCharsetLen, unsigned __int64 m_nPlainSpaceTotal, FILE* logFIle);
void Usage();
#endif
