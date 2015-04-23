#ifndef INCLUDE_NETWORK_H_
#define INCLUDE_NETWORK_H_

#include <stdlib.h>
#include <opencv2/core/core.hpp>

class NeuralNetwork {
 public:
    NeuralNetwork(const size_t& rnn_size, const size_t& layers_amount,
                  const double &init_val);
 private:
    size_t rnn_size_;
    size_t layers_amount_;
    std::vector<cv::Mat> encoder_layers_;
    std::vector<cv::Mat> decoder_layers_;
};

#endif  // INCLUDE_NETWORK_H_
