#include <iostream>

#include "../include/Generator.h"
#include "../include/Network.h"
#include "../include/SymbolManager.h"

const double NeuralNetwork::DX = 1E-4;

NeuralNetwork::NeuralNetwork(const size_t &rnn_size,
                             const size_t& layers_amount,
                             const double& init_val):
        rnn_size_(rnn_size), layers_amount_(layers_amount) {
    SymbolManager* symb_manager = SymbolManager::getInstance();
    Generator gen(strategy_);
    i2h = cv::Mat(symb_manager->getDictionarySize(), rnn_size, CV_64F);
    h2o = cv::Mat(rnn_size, symb_manager->getDictionarySize(), CV_64F);
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
    std::vector<cv::Mat> inputs, outputs;
    {
        int i = 0, j = 0;
        while (true) {
            if (i >= x.rows)
                break;
            if (std::abs(x.at<double>(i, 0)) < 1E-6) {
                ++i;
                continue;
            }
            j = i;
            while (x.at<double>(j, 0) > 1E-6)
                ++j;
            cv::Mat input = cv::Mat::zeros(j - i, 1, CV_64F);
            for (int h = i; h < j; ++h)
                input.at<double>(h - i, 0) = x.at<double>(h, 0);
            inputs.push_back(input);
            int y_pos = j;
            while (j < y.rows && y.at<double>(j, 0) > 1E-6)
                ++j;
            cv::Mat output = cv::Mat::zeros(j - y_pos, 1, CV_64F);
            for (int h = y_pos; h < j; ++h)
                output.at<double>(h - y_pos, 0) = y.at<double>(h, 0);
            outputs.push_back(output);
            i = j;
        }
    }
    for (int i = 0; i < i2h.rows; ++i) {
        for (int j = 0; j < i2h.cols; ++j) {
            cv::Mat i2h_left = i2h.clone();
            cv::Mat i2h_right = i2h.clone();
            i2h_left.at<double>(i, j) += DX;
            i2h_right.at<double>(i, j) -= DX;
            double left_prob = 0.0;
            double right_prob = 0.0;
            for (int h = 0; h < inputs.size(); ++h) {
                left_prob += estimateProbabilityOfOutput(inputs[h], outputs[h],
                                    W_encoder, W_decoder, b_encoder, b_decoder,
                                    i2h_left, h2o);
            }
            for (int h = 0; h < inputs.size(); ++h) {
                right_prob += estimateProbabilityOfOutput(inputs[h], outputs[h],
                                    W_encoder, W_decoder, b_encoder, b_decoder,
                                    i2h_right, h2o);
            }
            i2h_grad->at<double>(i, j) = (left_prob - right_prob) / (2.0 * DX);
        }
    }
    for (int i = 0; i < h2o.rows; ++i) {
        for (int j = 0; j < h2o.cols; ++j) {
            cv::Mat h2o_left = h2o.clone();
            cv::Mat h2o_right = h2o.clone();
            h2o_left.at<double>(i, j) += DX;
            h2o_right.at<double>(i, j) -= DX;
            double left_prob = 0.0;
            double right_prob = 0.0;
            for (int h = 0; h < inputs.size(); ++h) {
                left_prob += estimateProbabilityOfOutput(inputs[h], outputs[h],
                                    W_encoder, W_decoder, b_encoder, b_decoder,
                                    i2h, h2o_left);
            }
            for (int h = 0; h < inputs.size(); ++h) {
                right_prob += estimateProbabilityOfOutput(inputs[h], outputs[h],
                                    W_encoder, W_decoder, b_encoder, b_decoder,
                                    i2h, h2o_right);
            }
            h2o_grad->at<double>(i, j) = (left_prob - right_prob) / (2.0 * DX);
        }
    }
    for (int i = 0; i < W_encoder.rows; ++i) {
        for (int j = 0; j < W_encoder.cols; ++j) {
            cv::Mat W_enc_left = W_encoder.clone();
            cv::Mat W_enc_right = W_encoder.clone();
            W_enc_left.at<double>(i, j) += DX;
            W_enc_right.at<double>(i, j) -= DX;
            double left_prob = 0.0;
            double right_prob = 0.0;
            for (int h = 0; h < inputs.size(); ++h) {
                left_prob += estimateProbabilityOfOutput(inputs[h], outputs[h],
                                    W_enc_left, W_decoder, b_encoder, b_decoder,
                                    i2h, h2o);
            }
            for (int h = 0; h < inputs.size(); ++h) {
                right_prob += estimateProbabilityOfOutput(inputs[h], outputs[h],
                                    W_enc_right, W_decoder, b_encoder, b_decoder,
                                    i2h, h2o);
            }
            W_encoder_grad->at<double>(i, j) = (left_prob - right_prob) / (2.0 * DX);
        }
    }
    for (int i = 0; i < W_decoder.rows; ++i) {
        for (int j = 0; j < W_decoder.cols; ++j) {
            cv::Mat W_dec_left = W_decoder.clone();
            cv::Mat W_dec_right = W_decoder.clone();
            W_dec_left.at<double>(i, j) += DX;
            W_dec_right.at<double>(i, j) -= DX;
            double left_prob = 0.0;
            double right_prob = 0.0;
            for (int h = 0; h < inputs.size(); ++h) {
                left_prob += estimateProbabilityOfOutput(inputs[h], outputs[h],
                                    W_encoder, W_dec_left, b_encoder, b_decoder,
                                    i2h, h2o);
            }
            for (int h = 0; h < inputs.size(); ++h) {
                right_prob += estimateProbabilityOfOutput(inputs[h], outputs[h],
                                    W_encoder, W_dec_right, b_encoder, b_decoder,
                                    i2h, h2o);
            }
            W_decoder_grad->at<double>(i, j) = (left_prob - right_prob) / (2.0 * DX);
        }
    }
    for (int i = 0; i < b_encoder.rows; ++i) {
        for (int j = 0; j < b_encoder.cols; ++j) {
            cv::Mat b_enc_left = b_encoder.clone();
            cv::Mat b_enc_right = b_encoder.clone();
            b_enc_left.at<double>(i, j) += DX;
            b_enc_right.at<double>(i, j) -= DX;
            double left_prob = 0.0;
            double right_prob = 0.0;
            for (int h = 0; h < inputs.size(); ++h) {
                left_prob += estimateProbabilityOfOutput(inputs[h], outputs[h],
                                    W_encoder, W_decoder, b_enc_left, b_decoder,
                                    i2h, h2o);
            }
            for (int h = 0; h < inputs.size(); ++h) {
                right_prob += estimateProbabilityOfOutput(inputs[h], outputs[h],
                                    W_encoder, W_decoder, b_enc_right, b_decoder,
                                    i2h, h2o);
            }
            b_encoder_grad->at<double>(i, j) = (left_prob - right_prob) / (2.0 * DX);
        }
    }
    for (int i = 0; i < b_decoder.rows; ++i) {
        for (int j = 0; j < b_decoder.cols; ++j) {
            cv::Mat b_dec_left = b_decoder.clone();
            cv::Mat b_dec_right = b_decoder.clone();
            b_dec_left.at<double>(i, j) += DX;
            b_dec_right.at<double>(i, j) -= DX;
            double left_prob = 0.0;
            double right_prob = 0.0;
            for (int h = 0; h < inputs.size(); ++h) {
                left_prob += estimateProbabilityOfOutput(inputs[h], outputs[h],
                                    W_encoder, W_decoder, b_encoder, b_dec_left,
                                    i2h, h2o);
            }
            for (int h = 0; h < inputs.size(); ++h) {
                right_prob += estimateProbabilityOfOutput(inputs[h], outputs[h],
                                    W_encoder, W_decoder, b_encoder, b_dec_right,
                                    i2h, h2o);
            }
            b_decoder_grad->at<double>(i, j) = (left_prob - right_prob) / (2.0 * DX);
        }
    }
}

double NeuralNetwork::estimateProbabilityOfOutput(const cv::Mat &x, const cv::Mat &y,
                                                   const cv::Mat &W_enc, const cv::Mat &W_dec,
                                                   const cv::Mat &b_enc, const cv::Mat &b_dec,
                                                   const cv::Mat &itoh, const cv::Mat &htoo) {
    std::vector<cv::Mat> cur_h(layers_amount_ + 1,
                               cv::Mat::zeros(rnn_size_, 1, CV_64F));
    std::vector<cv::Mat> prev_h(layers_amount_ + 1,
                               cv::Mat::zeros(rnn_size_, 1, CV_64F));
    std::vector<cv::Mat> cur_c(layers_amount_ + 1,
                               cv::Mat::zeros(rnn_size_, 1, CV_64F));
    std::vector<cv::Mat> prev_c(layers_amount_ + 1,
                               cv::Mat::zeros(rnn_size_, 1, CV_64F));
    for (int i = 0; i < x.rows; ++i) {
        cur_h[0] = itoh.col(x.at<double>(i, 0) - 1).clone();
        for (size_t j = 1; j < layers_amount_ + 1; ++j) {
            cv::Mat arg(2 * rnn_size_, 1, CV_64F);
            for (int h = 0; h < rnn_size_; ++h)
                arg.at<double>(h, 0) = cur_h[j - 1].at<double>(h, 0);
            for (int h = 0; h < rnn_size_; ++h)
                arg.at<double>(h + rnn_size_, 0) = prev_h[j].at<double>(h, 0);
            cv::Mat out = W_enc * arg + b_enc;
            for (int h = 0; h < 3 * rnn_size_; ++h)
                out.at<double>(h, 0) = sigmoid(out.at<double>(h, 0));
            for (int h = 3 * rnn_size_; h < 4 * rnn_size_; ++h)
                out.at<double>(h, 0) = tanh(out.at<double>(h, 0));
            cv::Mat in(rnn_size_, 1, CV_64F), g(rnn_size_, 1, CV_64F),
                    f(rnn_size_, 1, CV_64F), o(rnn_size_, 1, CV_64F);
            for (int h = 0; h < 4 * rnn_size_; ++h) {
                if (h / rnn_size_ == 0)
                    in.at<double>(h % rnn_size_, 0) = out.at<double>(h, 0);
                else if (h / rnn_size_ == 1)
                    f.at<double>(h % rnn_size_, 0) = out.at<double>(h, 0);
                else if (h / rnn_size_ == 2)
                    o.at<double>(h % rnn_size_, 0) = out.at<double>(h, 0);
                else if (h / rnn_size_ == 0)
                    g.at<double>(h % rnn_size_, 0) = out.at<double>(h, 0);
            }
            cur_c[j] = f.mul(prev_c[j]) + in.mul(g);
            cv::Mat smoothed_c(rnn_size_, 1, CV_64F);
            for (int h = 0; h < rnn_size_; ++h)
                smoothed_c.at<double>(h, 0) = tanh(cur_c[j].at<double>(h, 0));
            cur_h[j] = o.mul(smoothed_c);
        }
    }
    cv::Mat encoded = cur_h[cur_h.size() - 1].clone();
    double res;
    cur_h = std::vector<cv::Mat>(layers_amount_ + 1,
                                 cv::Mat::zeros(rnn_size_, 1, CV_64F));
    prev_h = std::vector<cv::Mat>(layers_amount_ + 1,
                                  cv::Mat::zeros(rnn_size_, 1, CV_64F));
    cur_c = std::vector<cv::Mat>(layers_amount_ + 1,
                                 cv::Mat::zeros(rnn_size_, 1, CV_64F));
    prev_c = std::vector<cv::Mat>(layers_amount_ + 1,
                                  cv::Mat::zeros(rnn_size_, 1, CV_64F));
    SymbolManager* symb_manager = SymbolManager::getInstance();
    for (int i = 0; i < y.rows; ++i) {
        cur_h[0] = encoded.clone();
        for (size_t j = 1; j < layers_amount_ + 1; ++j) {
            cv::Mat arg(2 * rnn_size_, 1, CV_64F);
            for (int h = 0; h < rnn_size_; ++h)
                arg.at<double>(h, 0) = cur_h[j - 1].at<double>(h, 0);
            for (int h = 0; h < rnn_size_; ++h)
                arg.at<double>(h + rnn_size_, 0) = prev_h[j].at<double>(h, 0);
            cv::Mat out = W_dec * arg + b_dec;
            for (int h = 0; h < 3 * rnn_size_; ++h)
                out.at<double>(h, 0) = sigmoid(out.at<double>(h, 0));
            for (int h = 3 * rnn_size_; h < 4 * rnn_size_; ++h)
                out.at<double>(h, 0) = tanh(out.at<double>(h, 0));
            cv::Mat in(rnn_size_, 1, CV_64F), g(rnn_size_, 1, CV_64F),
                    f(rnn_size_, 1, CV_64F), o(rnn_size_, 1, CV_64F);
            for (int h = 0; h < 4 * rnn_size_; ++h) {
                if (h / rnn_size_ == 0)
                    in.at<double>(h % rnn_size_, 0) = out.at<double>(h, 0);
                else if (h / rnn_size_ == 1)
                    f.at<double>(h % rnn_size_, 0) = out.at<double>(h, 0);
                else if (h / rnn_size_ == 2)
                    o.at<double>(h % rnn_size_, 0) = out.at<double>(h, 0);
                else if (h / rnn_size_ == 0)
                    g.at<double>(h % rnn_size_, 0) = out.at<double>(h, 0);
            }
            cur_c[j] = f.mul(prev_c[j]) + in.mul(g);
            cv::Mat smoothed_c(rnn_size_, 1, CV_64F);
            for (int h = 0; h < rnn_size_; ++h)
                smoothed_c.at<double>(h, 0) = tanh(cur_c[j].at<double>(h, 0));
            cur_h[j] = o.mul(smoothed_c);
        }
        cv::Mat output = htoo * cur_h[cur_h.size() - 1];
        vectorToDistribution(output);
        double prob = output.at<double>(symb_manager->getIndexOfOutput(
                                            y.at<double>(i, 0)), 0);
        res -= log(prob);
    }
    return res;
}

void NeuralNetwork::vectorToDistribution(cv::Mat &x) {
    double sum = 0.0;
    for (int i = 0; i < x.rows; ++i) {
        x.at<double>(i, 0) = exp(x.at<double>(i, 0));
        sum += x.at<double>(i, 0);
    }
    for (int i = 0; i < x.rows; ++i)
        x.at<double>(i, 0) /= sum;
}

double NeuralNetwork::sigmoid(double x) {
    return 1 / (1 + exp(-x));
}
