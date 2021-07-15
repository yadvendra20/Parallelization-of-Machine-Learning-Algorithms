#include <iostream>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <cassert>
#include <cstring>
#include <cmath>
#include<omp.h>
 
 
#define PI 3.1415926535898
 
//The length of a single data
#define MAX_LINE 20
 //The length of the data set (training dataset 80% of total Dataset)
#define EIGEN_NUM 4
 int DATA_END=57120;
 //float dataSet[DATA_LEN * EIGEN_NUM]; //data set
float (*dataSet)=(float(*))malloc(sizeof(float)*DATA_END*EIGEN_NUM);  
 
 int dataLen;//The number of rows of the data set
 double maleNum=0;//Total number of males
 double femaleNum=0;//Total number of women
 int DATA_LEN=45696;
 //int DATA_END=57120;
int main(int argc, char **argv) {
 
	int i=0;
	int j=0;
 
	double start = omp_get_wtime( );
 
 
	 /************************Read file ************************ **/
 
	 char buf[MAX_LINE]; //Buffer
	 FILE *fp; //File pointer s
	 int len; //Number of line characters
 
		 //Read the file
		const char* fileLocation="data.csv";
		fp = fopen(fileLocation,"r");
		if(fp  == NULL)
		{
			perror("fp  == NULL");
			exit (1) ;
		}
 
		 //Read and write array line by line
		char *token;
		const char s[2] = ",";
		while(fgets(buf,MAX_LINE,fp) != NULL && i< DATA_END)
		{
			len = strlen(buf);
			 //Delete the newline
			buf[len-1] = '\0';
			 //Split string
			token = strtok(buf, s);
			 //Continue to split the string
			j = 0;
			while( token != NULL ) 
			{
				dataSet[i*EIGEN_NUM + j]=atof(token);
				token = strtok(NULL, s);
				j = j+1;
			 }
			i = i + 1;
		}
		dataLen=i;
		 printf("%d row 4 column data read completed\n",dataLen);
		fclose(fp);
 
		
 
		double readTime = omp_get_wtime( );
 
 
 
	 /************************Start OpenMP calculation ******************** **/
 
	int thread_count = strtol(argv[1],NULL,10);
 
	 /***********Calculate the Gaussian distribution***********/
	char *maenInf[6]={"maleLength","maleWeight","maleVC","femaleLength","femaleWeight","femaleVC"};
 
	double A,B,C,D,E,F,G;
	A=0;B=0;C=0;D=0;E=0;F=0;G=0;
 
	double sum[6]={0,0,0,0,0,0};
	double mean[6]={0,0,0,0,0,0};
 
 
#	pragma omp parallel for num_threads(thread_count) \
	reduction(+:maleNum) reduction(+:femaleNum) \
	reduction(+:A) reduction(+:B) reduction(+:C)\
	reduction(+:D) reduction(+:E) reduction(+:F)\
	shared(dataSet,DATA_LEN) private(i)
	for(i=0;i<DATA_LEN;i++)
	{
		if(dataSet[i*EIGEN_NUM]==1)
		{
			maleNum=maleNum+1;
			A+=dataSet[i*EIGEN_NUM+1];
			B+=dataSet[i*EIGEN_NUM+2];
			C+=dataSet[i*EIGEN_NUM+3];
		}
		else if(dataSet[i*EIGEN_NUM]==2)
		{
			femaleNum=femaleNum+1;
			D+=dataSet[i*EIGEN_NUM+1];
			E+=dataSet[i*EIGEN_NUM+2];
			F+=dataSet[i*EIGEN_NUM+3];
		}
		else
		{
			 printf("dataSet[%d]=%f,Gender is wrong\n",i*EIGEN_NUM,dataSet[i*EIGEN_NUM]);
		}
		//printf("sum[0]=%f \n",sum[0]);
		 //printf("The sum of the data in rows and 4 columns of %d is completed\n",i);
	}
#	pragma omp barrier
	sum[0]=A;
	sum[1]=B;
	sum[2]=C;
	sum[3]=D;
	sum[4]=E;
	sum[5]=F;
 
 
	//printf("maleNum=%.0f\nfemaleNum=%.0f\n",maleNum,femaleNum);
 
 
	/*for(i=0;i<6;i++)
	{
		printf("sum[%d]=%.0f\n",i,sum[i]);
	}*/
 
	 //Calculate the average
	for(i=0;i<6;i++)
	{
		if(i<3){mean[i]=sum[i]/maleNum;}
		if(i>2){mean[i]=sum[i]/femaleNum;}
		//printf("mean-%s = %.5f \n",maenInf[i],mean[i]);
	}
 
	 //Calculate the accumulation
	A=0;B=0;C=0;D=0;E=0;F=0;G=0;
	double Sigma[6]={0,0,0,0,0,0};
#	pragma omp parallel for num_threads(thread_count) default(none) \
	reduction(+:A) reduction(+:B) reduction(+:C)\
	reduction(+:D) reduction(+:E) reduction(+:F)\
	shared(dataSet,DATA_LEN,mean) private(i)
	for(i=0;i<DATA_LEN;i++)
	{
		if(dataSet[i*EIGEN_NUM]==1)
		{
			A+=pow(dataSet[i*EIGEN_NUM+1]-mean[0] , 2 );
			B+=pow(dataSet[i*EIGEN_NUM+2]-mean[1] , 2 );
			C+=pow(dataSet[i*EIGEN_NUM+3]-mean[2] , 2 );
		}
		else if(dataSet[i*EIGEN_NUM]==2)
		{
			D+=pow(dataSet[i*EIGEN_NUM+1]-mean[3] , 2 );
			E+=pow(dataSet[i*EIGEN_NUM+2]-mean[4] , 2 );
			F+=pow(dataSet[i*EIGEN_NUM+3]-mean[5] , 2 );
		}
		else
		{
			 printf("dataSet[i*EIGEN_NUM]=%f,Gender is wrong",dataSet[i*EIGEN_NUM]);
		}
	}
#	pragma omp barrier
	Sigma[0]=A;
	Sigma[1]=B;
	Sigma[2]=C;
	Sigma[3]=D;
	Sigma[4]=E;
	Sigma[5]=F;
 
 
	 //Calculate the standard deviation
	 double standardDeviation[6]; //standard deviation
	 double sexNum;//Number of each gender
	for(i=0;i<6;i++){
		if(i<3){sexNum=maleNum;}
		if(i>=3){sexNum=femaleNum;}
		standardDeviation[i]=sqrt(Sigma[i]/sexNum);
		//printf("Sigma[%d]=%f maleNum=%f",i,Sigma[i],sexNum);
		 //printf("%d standard deviation=%.5f\n",i,standardDeviation[i]);
		}
 
 
 
	 /*********** Naive Bayes & Accuracy Test ***********/
	 //Data set has vital capacity (VC), accuracy judgment
	float preSexID;
	float Right=0;
	float Error=0;
	 //Declare gender ID judgment function
	int sexIDResult(float height,float weight,float VC,double *mean,double *standardDeviation);
 
#	pragma omp parallel for num_threads(thread_count)  default(none) \
	reduction(+:Right) reduction(+:Error) \
	shared(dataSet,DATA_LEN,DATA_END,mean,standardDeviation) private(i,preSexID)
	for(i=DATA_LEN+1;i<DATA_END;i++){
		preSexID=sexIDResult(dataSet[i*EIGEN_NUM+1],dataSet[i*EIGEN_NUM+2],dataSet[i*EIGEN_NUM+3],mean,standardDeviation);
		if(dataSet[i*EIGEN_NUM]==preSexID){
			Right=Right+1;
		}
		else{
			Error=Error+1;
			 //printf("Forecast ID:%.0f Actual ID:%.0f \n",preSexID,receiveBuf[i*EIGEN_NUM]);
			 //printf("Sex:%.0f, height:%.2f, weight:%.2f, vital capacity:%.0f \n",receiveBuf[i*EIGEN_NUM],receiveBuf[i*EIGEN_NUM+1],receiveBuf[ i*EIGEN_NUM+2],receiveBuf[i*EIGEN_NUM+3]);
			}
	}
 
	printf("Right:%.0f\nError:%.0f\n",Right,Error);
	double accuracy  = Right/(Error+Right);
	printf("Accuracy:%f\n",accuracy);
 
	double end = omp_get_wtime( );
 
	//printf("start = %.16g\nend = %.16g\ndiff = %.16g\n", start, end, end - start);
	 printf("Overall time consuming = %.16f\n", end-start);
//	 printf("Read time = %.16f\n", readTime-start);
//	 printf("Calculation Time = %.16f\n", end-readTime);
 
	return 0;
}
 
 
 
 
 
 
 /*****************Function*****************/
 
 
 
 /***********Gaussian distribution function***********/
 //Sum
double getSum(float *data,int recDatalen,int sex,int column)
{
	double Sum=0;
	for(int i=0;i<(recDatalen/EIGEN_NUM);i++)
	{
		if(data[i*EIGEN_NUM]==sex){
			Sum=Sum+data[i*EIGEN_NUM+column];
		}
	}
	return Sum;
}
 
 //Find the accumulation of pow((data[i]-mean), 2)
double getSigma(float *data,int recDatalen,double mean,int sex,int column){
	double Sigma=0;
	for(int i=0;i<(recDatalen/EIGEN_NUM);i++){
		if(data[i*EIGEN_NUM]==sex){
			Sigma=Sigma+pow(data[i*EIGEN_NUM+column]-mean , 2 );
			//printf("sex=%d data[i]=%f mean=%f \n",sex,data[i*EIGEN_NUM+column],mean);
		}
	}
	return Sigma;
}
 
 
 
 /***********Naive Bayes function **********/
 
 //Calculate the probability p (feature column column = x | gender)
double getProbability(double x,int column,int sex,double mean,double standardDeviation)
{
	 double Probability; //Calculated probability
	double u = mean;
	double p = standardDeviation;
 
	 //High number distribution probability density function x: predictor variable u: sample mean p: standard deviation
	p=pow(p,2);
	Probability = (1 / (2*PI*p)) * exp( -pow((x-u),2) / (2*p) );
 
	 //printf("p(%s=%lf|gender=%s)=%.16lf\n",basicInfo[column],x,gender,Probability);
 
	return Probability;
}
 
 //Return the gender ID result
int sexIDResult(float height,float weight,float VC,double *mean,double *standardDeviation)
{
	 double maleP;//Male probability
	 double femaleP;//Female probability
	 double a=0.5; //The male and female ratio is 50% each
 
	maleP = a * getProbability(height,1,1,mean[0],standardDeviation[0]) * getProbability(weight,2,1,mean[1],standardDeviation[1]) 
		* getProbability(VC,3,1,mean[2],standardDeviation[2]);
 
	femaleP = a * getProbability(height,1,2,mean[3],standardDeviation[3]) * getProbability(weight,2,2,mean[4],standardDeviation[4]) 
		* getProbability(VC,3,2,mean[5],standardDeviation[5]);
 
	if(maleP > femaleP){return 1;}
	if(maleP < femaleP){return 2;}
	if(maleP == femaleP){return 0;}
}
