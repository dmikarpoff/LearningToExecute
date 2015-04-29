#ifndef INCLUDE_NETWORK_H_
#define INCLUDE_NETWORK_H_

#include <stdlib.h>
#include <opencv2/core/core.hpp>

#include "ILearningStrategy.h"

class NeuralNetwork {
 public:
    NeuralNetwork(const size_t& rnn_size, const size_t& layers_amount,
                  const double &init_val);
    void setStrategy(ILearningStrategy* strategy) {
        strategy_ = strategy;
    }
    void train(const size_t& batch_size, const size_t& train_length);
    void estimateGradientLogProbability(const cv::Mat &x, const cv::Mat &y,
                                        cv::Mat* i2h_grad, cv::Mat* h2o_grad,
                                        cv::Mat* W_encoder_grad,
                                        cv::Mat* W_decoder_grad,
                                        cv::Mat* b_encoder_grad,
                                        cv::Mat* b_decoder_grad);
    double estimateProbabilityOfOutput(const cv::Mat &x, const cv::Mat &y,
                                       const cv::Mat& W_enc, const cv::Mat& W_dec,
                                       const cv::Mat& b_enc, const cv::Mat& b_dec,
                                       const cv::Mat& itoh, const cv::Mat& htoo);
    double estimateDatasetProbability(const cv::Mat& x, const cv::Mat& y);
 private:
    static double sigmoid(double x);
    static void vectorToDistribution(cv::Mat& x);
    static const double DX;

    size_t rnn_size_;
    size_t layers_amount_;
    ILearningStrategy* strategy_;
    cv::Mat W_encoder, W_decoder, b_encoder, b_decoder;
    cv::Mat i2h;
    cv::Mat h2o;
};

#endif  // INCLUDE_NETWORK_H_
