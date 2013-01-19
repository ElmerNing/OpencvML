#pragma once
#include "SampleSet.h"
class Model
{
public:
	Model(const char* type_name CV_DEFAULT(CV_TYPE_NAME_ML_SVM));
	Model(const Model& model);
	~Model(void);

	//保存Model
	void Save(const char* filename);
	//从文件加载Model
	void Load(const char* filename);
	//清空Model
	void Clear();
	//训练Model
	void Train(const SampleSet& samples);
	void Train(const char* datsetPath);
	//识别
	void Predict(const SampleSet& samples, SampleSet& outError);
	void Predict(const char* datsetPath, const char* errDatsetPath = NULL);

	//设置参数,参数类型必须和当前Model的类型相对应
	void SetPara(const CvSVMParams& para);
	void SetPara(const CvEMParams& para);
	void SetPara(const CvBoostParams& para);
	void SetPara(const CvDTreeParams& para);
	void SetPara(const CvANN_MLP_TrainParams& para);
	void SetPara(const CvRTParams& para);
	void SetPara(const CvGBTreesParams& para);

private:
	void Train_svm(const SampleSet& samples);
	void Train_knn(const SampleSet& samples);
	void Train_nbayes(const SampleSet& samples);
	void Train_em(const SampleSet& samples);
	void Train_boosting(const SampleSet& samples);
	void Train_tree(const SampleSet& samples);
	void Train_mlp(const SampleSet& samples);
	void Train_rtrees(const SampleSet& samples);
	void Train_gbt(const SampleSet& samples);

	void Predict_svm(const SampleSet& samples, SampleSet& outError);
	void Predict_knn(const SampleSet& samples, SampleSet& outError);
	void Predict_nbayes(const SampleSet& samples, SampleSet& outError);
	void Predict_em(const SampleSet& samples, SampleSet& outError);
	void Predict_boosting(const SampleSet& samples, SampleSet& outError);
	void Predict_tree(const SampleSet& samples, SampleSet& outError);
	void Predict_mlp(const SampleSet& samples, SampleSet& outError);
	void Predict_rtrees(const SampleSet& samples, SampleSet& outError);
	void Predict_gbt(const SampleSet& samples, SampleSet& outError);

private:
	CvStatModel* m_pModel;
	void* m_trainPara;
	const char* m_type_name;

};

