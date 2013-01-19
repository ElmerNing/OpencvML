// datamining.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "SampleSet.h"
#include "Model.h"
int main(int argc, _TCHAR* argv[])
{
	int i;
	Model model(CV_TYPE_NAME_ML_ANN_MLP);
	model.SetPara(CvANN_MLP_TrainParams(cvTermCriteria(CV_TERMCRIT_ITER,100,0.001),
		CvANN_MLP_TrainParams::BACKPROP, 0.001));
	model.Train("_model.datset");

	model.Save("svm.model");
	model.Predict("_model.datset", "err.datset");

	scanf("%d", &i);

	model.Predict("_model.datset", "err.datset");

}

