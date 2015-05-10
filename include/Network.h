#ifndef INCLUDE_NETWORK_H_
#define INCLUDE_NETWORK_H_

#include <stdlib.h>
#include <opencv2/core/core.hpp>

#include "ILearningStrategy.h"

struct BackwardPropogationModule {
    cv::Mat grad_W;
    cv::Mat grad_b;
    cv::Mat grad_in1;
    cv::Mat grad_in2;
    cv::Mat grad_h2o;
};

class NeuralNetwork {
 public:
    NeuralNetwork(const size_t& rnn_size, const size_t& layers_amount,
                  const double &init_val);
    void setStrategy(ILearningStrategy* strategy) {
        strategy_ = strategy;
    }
    void train(const size_t& batch_size, const size_t& train_length);
//    void estimateGradientLikelihood(const std::vector<cv::Mat> &x,
//                                        const std::vector<cv::Mat> &y,
//                                        cv::Mat* i2h_grad, cv::Mat* h2o_grad,
//                                        cv::Mat* W_encoder_grad,
//                                        cv::Mat* W_decoder_grad,
//                                        cv::Mat* b_encoder_grad,
//                                        cv::Mat* b_decoder_grad);
    double estimateLikelihood(const cv::Mat &x, const cv::Mat &y,
                              const cv::Mat& W, cv::Mat& b,
                              const cv::Mat& input_gate1,
                              const cv::Mat& input_gate2,
                              const cv::Mat& h2o);
    double estimateDatasetLikelihood(const std::vector<cv::Mat>& x,
                                      const std::vector<cv::Mat>& y);
 private:
    static double sigmoid(double x);
    static void vectorToDistribution(cv::Mat& x);
    static const double DX;

    size_t rnn_size_;
    size_t layers_amount_;
    ILearningStrategy* strategy_;
    cv::Mat W_, b_;
    cv::Mat input_gate1_, input_gate2_;
    cv::Mat h2o_;
};

#endif  // INCLUDE_NETWORK_H_
