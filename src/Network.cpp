#include <iostream>

#include "../include/Generator.h"
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
    } while (strategy_->makeHarder());
    std::cout << "...End of training process" << std::endl;
}
