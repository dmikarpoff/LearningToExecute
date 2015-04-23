#include "../include/Network.h"

NeuralNetwork::NeuralNetwork(const size_t &rnn_size,
                             const size_t& layers_amount,
                             const double& init_val):
        rnn_size_(rnn_size), layers_amount_(layers_amount) {
    encoder_layers_.resize(layers_amount_);
    decoder_layers_.resize(layers_amount_);
    cv::RNG rng;
    for (size_t i = 0; i < encoder_layers_.size(); ++i) {
        encoder_layers_[i].create((int)rnn_size, 1, CV_64F);
        for (int j = 0; j < (int)rnn_size_; ++j)
            encoder_layers_[i].at<double>(j, 1) =
                                    rng.uniform(-init_val, init_val);
    }
    for (size_t i = 0; i < decoder_layers_.size(); ++i) {
        decoder_layers_[i].create((int)rnn_size, 1, CV_64F);
        for (int j = 0; j < (int)rnn_size_; ++j)
            decoder_layers_[i].at<double>(j, 1) =
                                    rng.uniform(-init_val, init_val);
    }
}
