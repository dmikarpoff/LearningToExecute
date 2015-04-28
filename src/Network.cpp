#include <iostream>

#include "../include/Generator.h"
#include "../include/Network.h"

NeuralNetwork::NeuralNetwork(const size_t &rnn_size,
                             const size_t& layers_amount,
                             const double& init_val):
        rnn_size_(rnn_size), layers_amount_(layers_amount) {
    encoder_layers_.resize(layers_amount_);
    decoder_layers_.resize(layers_amount_);
    for (size_t i = 0; i < encoder_layers_.size(); ++i) {
        encoder_layers_[i] = cv::Mat::zeros((int)rnn_size, 1, CV_64F);
    }
    for (size_t i = 0; i < decoder_layers_.size(); ++i) {
        decoder_layers_[i] = cv::Mat((int)rnn_size, 1, CV_64F);
    }
    Generator gen(strategy_);
    i2h = cv::Mat(gen.getDictionarySize(), rnn_size, CV_64F);
    h2o = cv::Mat(rnn_size, gen.getDictionarySize(), CV_64F);
    cv::RNG rng;
    for (int i = 0; i < i2h.rows; ++i)
        for (int j = 0; j < i2h.cols; ++j)
            i2h.at<double>(i, j) = rng.uniform(-init_val, init_val);
    for (int i = 0; i < h2o.rows; ++i)
        for (int j = 0; j < h2o.cols; ++j)
            h2o.at<double>(i, j) = rng.uniform(-init_val, init_val);
    W_encoder = cv::Mat(4 * rnn_size, 2 * rnn_size, CV_64F);
    W_decoder = cv::Mat(4 * rnn_size, 2 * rnn_size, CV_64F);
    b_encoder = cv::Mat(4 * rnn_size, 1, CV_64F);
    b_decoder = cv::Mat(4 * rnn_size, 1, CV_64F);
    for (int i = 0; i < 4 * rnn_size; ++i) {
        b_encoder.at<double>(i, 0) = rng.uniform(-init_val, init_val);
        b_decoder.at<double>(i, 0) = rng.uniform(-init_val, init_val);
        for (int j = 0; j < 2 * rnn_size; ++j) {
            W_encoder.at<double>(i, j) = rng.uniform(-init_val, init_val);
            W_decoder.at<double>(i, j) = rng.uniform(-init_val, init_val);
        }
    }
}

void NeuralNetwork::train(const size_t &batch_size, const size_t &train_length) {
    std::cout << std::endl;
    std::cout << "Starting training process..." << std::endl;
    std::cout << "\tBatch size = " << batch_size << std::endl;
    std::cout << "\tTraining length = " << train_length << std::endl;
    std::cout << std::endl;

    do {
        Generator generator(strategy_);
        size_t idx = 0;
        size_t batch_idx = 0;
        cv::Mat x = cv::Mat::ones(train_length, batch_size, CV_64F);
        cv::Mat y = cv::Mat::zeros(train_length, batch_size, CV_64F);
        size_t i = 0;
        std::cout << "Generating new input data" << std::endl;
        while (true) {
            TrainSample sample = generator.generate();
            if (idx + sample.input.size() >= train_length) {
                idx = 0;
                batch_idx++;
                if (batch_idx >= batch_size)
                    break;
            }
            for (size_t j = 0; j < sample.input.size(); ++j, ++idx) {
                if (j < sample.input.size())
                    x.at<double>(idx, batch_idx) = sample.input[j];
                if (j < sample.output.size())
                    y.at<double>(idx, batch_idx) = sample.output[j];
            }
            if (i <= 2) {
                std::cout << "\tInput:\n\t\t";
                for (size_t j = 0; j < sample.source_code.size(); ++j) {
                    if (sample.source_code[j] == '#')
                        std::cout << "\n\t\t";
                    else if (sample.source_code[j] == '@')
                        std::cout << "\n\tTarget:      ";
                    else
                        std::cout << (char)sample.source_code[j];
                }
                std::cout << std::endl;
                std::cout << "----------------------------" << std::endl;
                ++i;
            }
        }
        std::cout << "Generation done." << std::endl << std::endl;
        std::cout << "Maximizing aposteriori probability..." << std::endl;
        std::cout << "Aposteriori probablity maximized." << std::endl;
    } while (strategy_->makeHarder());
    std::cout << "...End of training process" << std::endl;
}

void NeuralNetwork::estimateGradientLogProbability(const cv::Mat &x,
                            const cv::Mat &y, cv::Mat *i2h_grad,
                            cv::Mat *h2o_grad, cv::Mat *W_encoder_grad,
                            cv::Mat *W_decoder_grad, cv::Mat *b_encoder_grad,
                            cv::Mat *b_decoder_grad) {

}
