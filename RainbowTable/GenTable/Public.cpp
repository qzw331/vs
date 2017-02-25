#include <math.h>
#include <iostream>
#ifdef _WIN32
	#pragma warning(disable : 4786)
#endif

#include "Public.h"

#ifdef _WIN32
	#include <windows.h>
#else
	#include <sys/sysinfo.h>
#endif


//////////////////////////////////////////////////////////////////////

unsigned int GetFileLen(FILE* file)
{
	unsigned int pos = ftell(file);
	fseek(file, 0, SEEK_END);
	unsigned int len = ftell(file);
	fseek(file, pos, SEEK_SET);

	return len;
}

string TrimString(string s)
{
	while (s.size() > 0)
	{
		if (s[0] == ' ' || s[0] == '\t')
			s = s.substr(1);
		else
			break;
	}

	while (s.size() > 0)
	{
		if (s[s.size() - 1] == ' ' || s[s.size() - 1] == '\t')
			s = s.substr(0, s.size() - 1);
		else
			break;
	}

	return s;
}

bool ReadLinesFromFile(string sPathName, vector<string>& vLine)
{
	vLine.clear();

	FILE* file = fopen(sPathName.c_str(), "rb");
	if (file != NULL)
	{
		unsigned int len = GetFileLen(file);
		char* data = new char[len + 1];
		fread(data, 1, len, file);
		data[len] = '\0';
		string content = data;
		content += "\n";
		delete data;

		int i;
		for (i = 0; i < content.size(); i++)
		{
			if (content[i] == '\r')
				content[i] = '\n';
		}

		int n;
		while ((n = content.find("\n", 0)) != -1)
		{
			string line = content.substr(0, n);
			line = TrimString(line);
			if (line != "")
				vLine.push_back(line);
			content = content.substr(n + 1);
		}

		fclose(file);
	}
	else
		return false;

	return true;
}

bool SeperateString(string s, string sSeperator, vector<string>& vPart)
{
	vPart.clear();

	int i;
	for (i = 0; i < sSeperator.size(); i++)
	{
		int n = s.find(sSeperator[i]);
		if (n != -1)
		{
			vPart.push_back(s.substr(0, n));
			s = s.substr(n + 1);
		}
		else
			return false;
	}
	vPart.push_back(s);

	return true;
}

string uint64tostr(uint64 n)
{
	char str[32];

#ifdef _WIN32
	sprintf(str, "%I64u", n);
#else
	sprintf(str, "%llu", n);
#endif

	return str;
}

string uint64tohexstr(uint64 n)
{
	char str[32];

#ifdef _WIN32
	sprintf(str, "%016I64x", n);
#else
	sprintf(str, "%016llx", n);
#endif

	return str;
}

string HexToStr(const unsigned char* pData, int nLen)
{
	string sRet;
	int i;
	for (i = 0; i < nLen; i++)
	{
		char szByte[3];
		sprintf(szByte, "%02x", pData[i]);
		sRet += szByte;
	}

	return sRet;
}


//计算参数说明：
//	给出彩虹表总的链数M、希望达到的覆盖率P、已知的明文空间N
//	则可计算出彩虹表的表数n、每张表所含链数m、每条链长度t
//	计算公式：
//	n = [-ln(1-P)/2]向上取整
//	m = M/n
//	t = -N*ln(1-P)/M
//	应用多张表目的是减少重复的链
//	不同表之间的区别在于HashToIndex函数不一样
//	HashToIndex 函数算法含有两个可变参数，分别为GPUReduceOffset和pos，其中pos该环节在链中的位置，而GPUReduceOffset=表索引*65536
//	相同表的GPUReduceOffset是一样的，不同表的GPUReduceOffset不一样
//	这样，如果两条链中相同的位置（即pos参数一样）产生了相同的hash结果，若这两条链分别在不同的表中，则会产生不一样的Index（因为GPUReduceOffset不一样）
bool Calculator(double AllTableSize, double CoverProbality, int &nNumberTable, unsigned __int64 &RainbowChainOfEachTable,
	unsigned __int64 &AllChainNumber, unsigned __int64 &RainbowChainLength, int nPlainLenMin, int nPlainLenMax,
	int m_nPlainCharsetLen, unsigned __int64 m_nPlainSpaceTotal, FILE* logFIle)
{ 
	if(m_nPlainCharsetLen<9)
	{
		printf("The charset length is wrong!\n"); 
		return 0;
	}
	nNumberTable = ceil(fabs(log(1 - CoverProbality) / 2));
	AllChainNumber = AllTableSize*oneTera / sizeof(RainbowChain);
	RainbowChainOfEachTable = AllChainNumber / nNumberTable;
	RainbowChainLength = fabs(m_nPlainSpaceTotal*log(1 - CoverProbality) / AllChainNumber);

	////////////////////////////////////////////////////
	//RainbowChainLength = 258312;
	/////////////////////////////////////////////////
	if (m_nPlainSpaceTotal < AllChainNumber)
	{   printf("-----------------------------Warning!---------------------------------\n");
		printf("Warning: the disk space you given is bigger than the key space size!\n");
		printf("Maybe the best way to crack is using brute-force instead of rain bow table!\n");
		printf("-----------------------------Warning over!---------------------------------\n");

		 fprintf(logFIle,"-----------------------------Warning!---------------------------------\n");
		fprintf(logFIle,"Warning: the disk space you given is bigger than the key space size!\n");
		fprintf(logFIle,"Maybe the best way to crack is using brute-force instead of rain bow table!\n");
		fprintf(logFIle,"-----------------------------Warning over!---------------------------------\n");
		return false;
	}

	printf("-----------------------------Report---------------------------------\n");
	printf("字符集长度%d, 密码最小长度%d ,密码最大长度%d,需要的覆盖率p=%lf。 \n", m_nPlainCharsetLen, nPlainLenMin, nPlainLenMax, CoverProbality);
    printf("表数 %d,全部的链数=%I64d，链长= %I64d，每张表链数=%I64d \n",nNumberTable, AllChainNumber,RainbowChainLength,RainbowChainOfEachTable);
    printf("-----------------------------Report  Over---------------------------\n");

	fprintf(logFIle,"-----------------------------Report---------------------------------\n");
	fprintf(logFIle, "字符集长度%d, 密码最小长度%d ,密码最大长度%d,需要的覆盖率p=%lf。 \n", m_nPlainCharsetLen, nPlainLenMin, nPlainLenMax, CoverProbality);
	fprintf(logFIle, "表数 %d, 全部的链数 = %I64d，链长 = %I64d，每张表链数=%I64d \n", nNumberTable, AllChainNumber, RainbowChainLength, RainbowChainOfEachTable);
    fprintf(logFIle,"-----------------------------Report  Over---------------------------\n");
	return true;
}


 void Usage()
{
	printf("———————————————————————————————\n");	
	printf("usage: GenTable hash_algorithm \n");
	printf("		plain_charset \n \t\tplain_len_min \n \t\tplain_len_max \n");
	printf("		AllTableSize \n");
	printf("		CoverProbality  \n");
	printf("		Table_Index \n");
	printf("		file_title_suffix\n\n");

	
	printf("hash_algorithm:		available: md5\n");
	printf("plain_charset:		use any charset name in charset.txt here\n");
	printf("plain_len_min:		min length of the plaintext\n");
	printf("plain_len_max:		max length of the plaintext\n");
	printf("AllTableSize:		the size (Tera) of all tables to be generated \n");
	printf("CoverProbality:		wanted cover probality\n");	
	printf("Table_Index:		the index of the first subtable to be generated\n");	
	printf("file_title_suffix:	the string appended to the file title\n\n");
	printf("example: \n");
	
	printf("GenTable md5 alpha 1 7 0.1 0.95 2 test   \n");
	printf("GenTable md5 byte 4 4 2 0.99 1 test   \n");
	 printf("———————————————————————————————\n");	
}
