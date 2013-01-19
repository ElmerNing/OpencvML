#pragma once
class SampleSet
{
public:
	SampleSet(void);
	~SampleSet(void);
	
	const cv::Mat& Samples() const { return m_samples; }
	const cv::Mat& Labels() const { return m_labels; }
	int N() const { return m_samples.rows; }
	int Dim() const { return m_samples.cols; }
	vector<float> Classes() const;

	cv::Mat GetSampleAt(int n) const {return m_samples.row(n);}
	float GetLabelAt(int n) const {return m_labels.at<float>(n, 0);}

	bool Read(const char* filename);
	void Write(const char* filename);

	void Add(cv::Mat sample, float label);

private:
	cv::Mat m_samples;
	cv::Mat m_labels;
};

