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


//�������˵����
//	�����ʺ���ܵ�����M��ϣ���ﵽ�ĸ�����P����֪�����Ŀռ�N
//	��ɼ�����ʺ��ı���n��ÿ�ű���������m��ÿ��������t
//	���㹫ʽ��
//	n = [-ln(1-P)/2]����ȡ��
//	m = M/n
//	t = -N*ln(1-P)/M
//	Ӧ�ö��ű�Ŀ���Ǽ����ظ�����
//	��ͬ��֮�����������HashToIndex������һ��
//	HashToIndex �����㷨���������ɱ�������ֱ�ΪGPUReduceOffset��pos������pos�û��������е�λ�ã���GPUReduceOffset=������*65536
//	��ͬ���GPUReduceOffset��һ���ģ���ͬ���GPUReduceOffset��һ��
//	�������������������ͬ��λ�ã���pos����һ������������ͬ��hash����������������ֱ��ڲ�ͬ�ı��У���������һ����Index����ΪGPUReduceOffset��һ����
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
	printf("�ַ�������%d, ������С����%d ,������󳤶�%d,��Ҫ�ĸ�����p=%lf�� \n", m_nPlainCharsetLen, nPlainLenMin, nPlainLenMax, CoverProbality);
    printf("���� %d,ȫ��������=%I64d������= %I64d��ÿ�ű�����=%I64d \n",nNumberTable, AllChainNumber,RainbowChainLength,RainbowChainOfEachTable);
    printf("-----------------------------Report  Over---------------------------\n");

	fprintf(logFIle,"-----------------------------Report---------------------------------\n");
	fprintf(logFIle, "�ַ�������%d, ������С����%d ,������󳤶�%d,��Ҫ�ĸ�����p=%lf�� \n", m_nPlainCharsetLen, nPlainLenMin, nPlainLenMax, CoverProbality);
	fprintf(logFIle, "���� %d, ȫ�������� = %I64d������ = %I64d��ÿ�ű�����=%I64d \n", nNumberTable, AllChainNumber, RainbowChainLength, RainbowChainOfEachTable);
    fprintf(logFIle,"-----------------------------Report  Over---------------------------\n");
	return true;
}


 void Usage()
{
	printf("��������������������������������������������������������������\n");	
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
	 printf("��������������������������������������������������������������\n");	
}
