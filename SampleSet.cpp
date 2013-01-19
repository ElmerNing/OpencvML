#include "StdAfx.h"
#include "SampleSet.h"

static void remove_space(string& str){ 
	if (str == "")
		return;
	string buff(str); 
	char space = ' '; 
	str.assign(buff.begin() + buff.find_first_not_of(space), 
		buff.begin() + buff.find_last_not_of(space) + 1); 
}

SampleSet::SampleSet(void)
{
}

SampleSet::~SampleSet(void)
{
}

bool SampleSet::Read( const char* filename )
{
	ifstream ifs(filename);
	if (!ifs.is_open())
		return false;

	string line;
	vector<vector<float>> samples;

	do 
	{
		vector<float> sample;
		if ( ifs.eof() )
			break;

		getline(ifs, line);
		remove_space(line);
		stringstream ss(line);
		if (line.size() < 4)
			break;
		
		do 
		{
			float x = 0;
			ss>>x;
			sample.push_back(x);
		} while (!ss.eof());
		samples.push_back(sample);
	} while(true);

	int n = samples.size();
	int dim = 0;
	for (vector<vector<float>>::iterator it = samples.begin(); it != samples.end(); it++)
	{
		if(dim != 0)
			dim = min(dim, (int)it->size() - 1);
		else
			dim = (int)it->size() - 1;
	}

	if (n <= 0 || dim <=0)
		return false;

	m_samples.create(n, dim, CV_32F);
	m_labels.create(n,1,CV_32F);
	for (int i=0; i < n; i++)
	{
		m_labels.at<float>(i,0) = samples[i][0];;
		for (int j=0; j < dim; j++)
		{
			m_samples.at<float>(i, j) = samples[i][j+1];
		}
	}
	return true;
}

void SampleSet::Write( const char* filename )
{
	ofstream ofs(filename);
	double temp = 0.803921568627451;
	ofs.precision(20);
	for (int i=0; i<N(); i++)
	{
		ofs<<GetLabelAt(i);
		for (int d=0; d<Dim(); d++)
		{
			temp = m_samples.at<float>(i,d);
			ofs<<" "<<temp;
		}
	}
	ofs.close();
}

void SampleSet::Add( cv::Mat sample, float label )
{
	m_samples.push_back(sample);
	m_labels.push_back(label);
}

vector<float> SampleSet::Classes() const
{
	vector<float> classes;
	for (int n=0; n<this->N(); n++)
	{
		bool isExist = false;
		float label = this->GetLabelAt(n);
		for (int i=0; i<(int)classes.size(); i++)
		{
			if (label == classes[i])
			{
				isExist = true;
				break;
			}
		}
		if (isExist == false)
			classes.push_back(label);
	}
	sort(classes.begin(), classes.end());
	return classes;
}