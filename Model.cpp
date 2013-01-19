#include "StdAfx.h"
#include "Model.h"


Model::Model(const char* type_name)
:m_pModel(NULL)
,m_trainPara(NULL)
{
	m_type_name = type_name;
	if (!strcmp(type_name, CV_TYPE_NAME_ML_SVM))
	{
		m_pModel = new CvSVM();
		//构造初始化默认参数
		CvSVMParams* para = new CvSVMParams();
		para->svm_type = CvSVM::C_SVC;
		para->kernel_type = CvSVM::RBF;
		para->C = 27.68;
		para->gamma = 0.023;
		m_trainPara = para;
	}
	else if (!strcmp(type_name, CV_TYPE_NAME_ML_KNN))
	{
		m_pModel = new CvKNearest();
		m_trainPara = NULL;
	}
	else if (!strcmp(type_name, CV_TYPE_NAME_ML_NBAYES))
	{
		m_pModel = new CvNormalBayesClassifier();
		m_trainPara = NULL;
	}
	else if (!strcmp(type_name, CV_TYPE_NAME_ML_EM))
	{
		m_pModel = new CvEM();
		m_trainPara = new CvEMParams();
	}
	else if (!strcmp(type_name, CV_TYPE_NAME_ML_BOOSTING))
	{
		m_pModel = new CvBoost();
		m_trainPara = new CvBoostParams();
	}
	else if (!strcmp(type_name, CV_TYPE_NAME_ML_TREE))
	{
		m_pModel = new CvDTree();
		m_trainPara = new CvDTreeParams();
	}
	else if (!strcmp(type_name, CV_TYPE_NAME_ML_ANN_MLP))
	{
		m_pModel = new CvANN_MLP();
		m_trainPara = new CvANN_MLP_TrainParams(cvTermCriteria(CV_TERMCRIT_ITER,30000,0.001),CvANN_MLP_TrainParams::BACKPROP, 0.001);
	}
	else if (!strcmp(type_name, CV_TYPE_NAME_ML_RTREES))
	{
		m_pModel = new CvRTrees();
		m_trainPara = new CvRTParams();
	}
	else if (!strcmp(type_name, CV_TYPE_NAME_ML_GBT))
	{
		m_pModel = new CvGBTrees();
		m_trainPara = new CvGBTreesParams(CvGBTrees::DEVIANCE_LOSS, 100, 0.1f, 0.1f, 3, false );
	}
	else
	{
		cerr<<type_name<<"is not supported"<<endl;
		exit(1);
		//m_pModel = NULL;
		//m_trainPara = NULL;
	}
}

Model::Model( const Model& model )
{
	m_pModel = new CvStatModel(*model.m_pModel);
}

Model::~Model(void)
{
	if (NULL != m_pModel)
	{
		delete m_pModel;
		m_pModel = NULL;
	}
	if (NULL != m_trainPara)
	{
		delete m_trainPara;
		m_trainPara = NULL;
	}
}

void Model::Save( const char* filename )
{
	if ( NULL == m_pModel)
		return;
	m_pModel->save(filename);
}

void Model::Load( const char* filename )
{
	if (NULL == m_pModel)
	{
		//判断格式,并加载
		assert(m_pModel);
		//以后修改为自动判断格式
		m_pModel = new CvStatModel();
		m_pModel->load(filename);
	}
	m_pModel->load(filename);
}

void Model::Train( const SampleSet& samples )
{
	if (!strcmp(m_type_name, CV_TYPE_NAME_ML_SVM))
		Train_svm(samples);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_KNN))
		Train_knn(samples);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_NBAYES))
		Train_nbayes(samples);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_EM))
		Train_em(samples);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_BOOSTING))
		Train_boosting(samples);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_TREE))
		Train_tree(samples);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_ANN_MLP))
		Train_mlp(samples);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_RTREES))
		Train_rtrees(samples);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_GBT))
		Train_gbt(samples);
	else
		m_pModel = NULL;
}

void Model::Train( const char* datsetPath )
{
	SampleSet samples;
	samples.Read(datsetPath);
	Train(samples);
}

void Model::Predict( const SampleSet& samples, SampleSet& outError)
{
	if (!strcmp(m_type_name, CV_TYPE_NAME_ML_SVM))
		Predict_svm(samples, outError);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_KNN))
		Predict_knn(samples, outError);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_NBAYES))
		Predict_nbayes(samples, outError);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_EM))
		Predict_em(samples, outError);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_BOOSTING))
		Predict_boosting(samples, outError);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_TREE))
		Predict_tree(samples, outError);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_ANN_MLP))
		Predict_mlp(samples, outError);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_RTREES))
		Predict_rtrees(samples, outError);
	else if (!strcmp(m_type_name, CV_TYPE_NAME_ML_GBT))
		Predict_gbt(samples, outError);
	else
		m_pModel = NULL;
}

void Model::Predict( const char* datsetPath, const char* errDatsetPath )
{
	SampleSet samples, errorSample;
	samples.Read(datsetPath);
	
	Predict(samples, errorSample);
	
	if (errDatsetPath != NULL)
		errorSample.Write(errDatsetPath);
}

#pragma region SetPara
void Model::SetPara( const CvSVMParams& para )
{
	assert(0 == strcmp(m_type_name, CV_TYPE_NAME_ML_SVM));
	CvSVMParams* p = (CvSVMParams*)m_trainPara;
	if (NULL != p)
		*p = para;
}

void Model::SetPara( const CvEMParams& para )
{
	assert(0 == strcmp(m_type_name, CV_TYPE_NAME_ML_EM));
	CvEMParams* p = (CvEMParams*)m_trainPara;
	if (NULL != p)
		*p = para;
}

void Model::SetPara( const CvBoostParams& para )
{
	assert(0 == strcmp(m_type_name, CV_TYPE_NAME_ML_BOOSTING));
	CvBoostParams* p = (CvBoostParams*)m_trainPara;
	if (NULL != p)
		*p = para;
}

void Model::SetPara( const CvDTreeParams& para )
{
	assert(0 == strcmp(m_type_name, CV_TYPE_NAME_ML_TREE));
	CvDTreeParams* p = (CvDTreeParams*)m_trainPara;
	if (NULL != p)
		*p = para;
}

void Model::SetPara( const CvANN_MLP_TrainParams& para )
{
	assert(0 == strcmp(m_type_name, CV_TYPE_NAME_ML_ANN_MLP));
	CvANN_MLP_TrainParams* p = (CvANN_MLP_TrainParams*)m_trainPara;
	if (NULL != p)
		*p = para;
}

void Model::SetPara( const CvRTParams& para )
{
	assert(0 == strcmp(m_type_name, CV_TYPE_NAME_ML_RTREES));
	CvRTParams* p = (CvRTParams*)m_trainPara;
	if (NULL != p)
		*p = para;
}

void Model::SetPara( const CvGBTreesParams& para )
{
	assert(0 == strcmp(m_type_name, CV_TYPE_NAME_ML_GBT));
	CvGBTreesParams* p = (CvGBTreesParams*)m_trainPara;
	if (NULL != p)
		*p = para;
}
#pragma endregion SetPara

#pragma region Train
void Model::Train_svm( const SampleSet& samples )
{
	CvSVM* model = (CvSVM*)m_pModel;
	CvSVMParams* para = (CvSVMParams*)m_trainPara;
	model->train(samples.Samples(), samples.Labels(), cv::Mat(),cv::Mat(), *para);
}

void Model::Train_knn( const SampleSet& samples )
{
	CvKNearest* model = (CvKNearest*)m_pModel;
	//void* para = (void*)m_trainPara;
	model->train(samples.Samples(), samples.Labels());
}

void Model::Train_nbayes( const SampleSet& samples )
{
	CvNormalBayesClassifier* model = (CvNormalBayesClassifier*)m_pModel;
	//void* para = (void*)m_trainPara;
	model->train(samples.Samples(), samples.Labels());
}

void Model::Train_em( const SampleSet& samples )
{
	CvEM* model = (CvEM*)m_pModel;
	CvEMParams* para = (CvEMParams*)m_trainPara;
	para->nclusters = samples.Classes().size();
	model->train(samples.Samples());
}

void Model::Train_boosting( const SampleSet& samples )
{
	CvBoost*model = (CvBoost*)m_pModel;
	CvBoostParams* para = (CvBoostParams*)m_trainPara;
	model->train(samples.Samples(), CV_ROW_SAMPLE, samples.Labels(), 
		cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), *para);
}

void Model::Train_tree( const SampleSet& samples )
{
	CvDTree* model = (CvDTree*)m_pModel;
	CvDTreeParams* para = (CvDTreeParams*)m_trainPara;
	model->train(samples.Samples(), CV_ROW_SAMPLE, samples.Labels(), 
		cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), *para);
}

void Model::Train_mlp( const SampleSet& samples )
{
	CvANN_MLP* model = (CvANN_MLP*)m_pModel;
	CvANN_MLP_TrainParams* para = (CvANN_MLP_TrainParams*)m_trainPara;
	
	int dim = samples.Dim();
	vector<float> classes = samples.Classes();
	cv::Mat layerSize = (cv::Mat_<int>(1, 3) << dim, 100, classes.size());
	model->create(layerSize);

	cv::Mat newLaybels = cv::Mat::zeros(samples.N(), classes.size(), CV_32F);
	for (int n=0; n<samples.N(); n++)
	{
		int label = samples.GetLabelAt(n);
		for (int c=0; c<classes.size(); c++)
		{
			if (label == classes[c])
				newLaybels.at<float>(n, c) = 1.0f;
		}
	}
	model->train(samples.Samples(), newLaybels, cv::Mat::ones(samples.N(), 1, CV_32F), cv::Mat(), *para);
}

void Model::Train_rtrees( const SampleSet& samples )
{
	CvRTrees* model = (CvRTrees*)m_pModel;
	CvRTParams* para = (CvRTParams*)m_trainPara;
	model->train(samples.Samples(), CV_ROW_SAMPLE, samples.Labels(), 
		cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), *para);
}

void Model::Train_gbt( const SampleSet& samples )
{
	CvGBTrees* model = (CvGBTrees*)m_pModel;
	CvGBTreesParams* para = (CvGBTreesParams*)m_trainPara;
	model->train(samples.Samples(), CV_ROW_SAMPLE, samples.Labels(), 
		cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), *para);
}
#pragma endregion Train

#pragma region Predict
void Model::Predict_svm( const SampleSet& samples, SampleSet& outError )
{
	int true_resp = 0;
	CvSVM *model = (CvSVM*)m_pModel;

	for (int i = 0; i < samples.N(); i++)
	{
		float ret = model->predict(samples.GetSampleAt(i));
		if (ret != samples.GetLabelAt(i))
		{
			outError.Add(samples.GetSampleAt(i), samples.GetLabelAt(i));
		}
		else
		{
			true_resp++;
		}
	}
	printf("%d %d",samples.N(), true_resp);
}

void Model::Predict_knn( const SampleSet& samples, SampleSet& outError )
{
	int true_resp = 0;
	CvKNearest *model = (CvKNearest*)m_pModel;
	cv::Mat result;
	model->find_nearest(samples.Samples(), 1, &result);
	for (int i = 0; i < samples.N(); i++)
	{
		if (result.at<float>(i) != samples.GetLabelAt(i))
		{
			outError.Add(samples.GetSampleAt(i), samples.GetLabelAt(i));
		}
		else
		{
			true_resp++;
		}
	}
	printf("%d %d",samples.N(), true_resp);
}

void Model::Predict_nbayes( const SampleSet& samples, SampleSet& outError )
{
	int true_resp = 0;
	CvNormalBayesClassifier* model = (CvNormalBayesClassifier*)m_pModel;
	cv::Mat result;
	model->predict(samples.Samples(),&result);
	for (int i = 0; i < samples.N(); i++)
	{
		if (result.at<float>(i) != samples.GetLabelAt(i))
		{
			outError.Add(samples.GetSampleAt(i), samples.GetLabelAt(i));
		}
		else
		{
			true_resp++;
		}
	}
	printf("%d %d",samples.N(), true_resp);
}

void Model::Predict_em( const SampleSet& samples, SampleSet& outError )
{
	int true_resp = 0;
	CvEM *model = (CvEM*)m_pModel;

	for (int i = 0; i < samples.N(); i++)
	{
		float ret = model->predict(samples.GetSampleAt(i));
		if (ret != samples.GetLabelAt(i))
		{
			outError.Add(samples.GetSampleAt(i), samples.GetLabelAt(i));
		}
		else
		{
			true_resp++;
		}
	}
	printf("%d %d",samples.N(), true_resp);
}

//目前只能进行两分类的识别，否则会抛出异常
void Model::Predict_boosting( const SampleSet& samples, SampleSet& outError )
{
	int true_resp = 0;
	CvBoost *model = (CvBoost*)m_pModel;

	for (int i = 0; i < samples.N(); i++)
	{
		float ret = model->predict(samples.GetSampleAt(i), cv::Mat(), cv::Range::all());
		if (ret != samples.GetLabelAt(i))
		{
			outError.Add(samples.GetSampleAt(i), samples.GetLabelAt(i));
		}
		else
		{
			true_resp++;
		}
	}
	printf("%d %d",samples.N(), true_resp);
}

void Model::Predict_tree( const SampleSet& samples, SampleSet& outError )
{
	int true_resp = 0;
	CvDTree *model = (CvDTree*)m_pModel;

	for (int i = 0; i < samples.N(); i++)
	{
		CvDTreeNode *pnode;
		pnode = model->predict(samples.GetSampleAt(i), cv::Mat());
		if (pnode->value != samples.GetLabelAt(i))
		{
			outError.Add(samples.GetSampleAt(i), samples.GetLabelAt(i));
		}
		else
		{
			true_resp++;
		}
	}
	printf("%d %d",samples.N(), true_resp);
}

void Model::Predict_mlp( const SampleSet& samples, SampleSet& outError )
{
	
	int true_resp = 0;
	CvANN_MLP *model = (CvANN_MLP*)m_pModel;
	cv::Mat result;
	float temp[40];

	model->predict(samples.Samples(), result);

	for (int i = 0; i < samples.N(); i++)
	{
		float maxcol = -1;
		int index = -1;
		for (int j = 0; j < result.cols; j++)
		{
			if (result.at<float>(i,j) > maxcol)
			{
				maxcol = result.at<float>(i,j);
				index = j;
			}
 		}
		float label = samples.Classes()[index];
		if (label != samples.GetLabelAt(i))
		{
			outError.Add(samples.GetSampleAt(i), samples.GetLabelAt(i));
		}
		else
		{
			true_resp++;
		}
	}
	printf("%d %d",samples.N(), true_resp);
}

void Model::Predict_rtrees( const SampleSet& samples, SampleSet& outError )
{	
	int true_resp = 0;
	CvRTrees *model = (CvRTrees*)m_pModel;
	
	for (int i = 0; i < samples.N(); i++)
	{
		float ret = model->predict(samples.GetSampleAt(i), cv::Mat());
		if (ret != samples.GetLabelAt(i))
		{
			outError.Add(samples.GetSampleAt(i), samples.GetLabelAt(i));
		}
		else
		{
			true_resp++;
		}
	}
	printf("%d %d",samples.N(), true_resp);
}

void Model::Predict_gbt( const SampleSet& samples, SampleSet& outError )
{
	int true_resp = 0;
	CvGBTrees *model = (CvGBTrees*)m_pModel;
	for (int i = 0; i < samples.N(); i++)
	{
		float ret;
		ret = model->predict(samples.GetSampleAt(i), cv::Mat(), cv::Range::all());
		if (ret != samples.GetLabelAt(i))
		{
			outError.Add(samples.GetSampleAt(i), samples.GetLabelAt(i));
		}
		else
		{
			true_resp++;
		}
	}
	printf("%d %d",samples.N(), true_resp);
}
#pragma endregion Predict





