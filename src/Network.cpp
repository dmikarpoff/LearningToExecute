#include <omp.h>

#include <iostream>

#include "../include/Generator.h"
#include "../include/Network.h"
#include "../include/SymbolManager.h"

const double NeuralNetwork::DX = 1E-4;

#define USE_OMP

NeuralNetwork::NeuralNetwork(const size_t &rnn_size,
                             const size_t& layers_amount,
                             const double& init_val):
        rnn_size_(rnn_size), layers_amount_(layers_amount) {
    SymbolManager* symb_manager = SymbolManager::getInstance();
    input_gate1_ = cv::Mat(symb_manager->getDictionarySize(), rnn_size, CV_64F);
    input_gate2_ = cv::Mat(symb_manager->getOutDictSize(), rnn_size, CV_64F);
    h2o_ = cv::Mat(symb_manager->getOutDictSize(), rnn_size, CV_64F);
    cv::RNG rng;
    for (int i = 0; i < input_gate1_.rows; ++i)
        for (int j = 0; j < input_gate1_.cols; ++j)
            input_gate1_.at<double>(i, j) = rng.uniform(-init_val, init_val);
    for (int i = 0; i < h2o_.rows; ++i)
        for (int j = 0; j < h2o_.cols; ++j) {
            h2o_.at<double>(i, j) = rng.uniform(-init_val, init_val);
            input_gate2_.at<double>(i, j) = rng.uniform(-init_val, init_val);
        }
    W_ = cv::Mat(4 * rnn_size, 2 * rnn_size, CV_64F);
    b_ = cv::Mat(4 * rnn_size, 1, CV_64F);
    for (int i = 0; i < 4 * rnn_size; ++i) {
        b_.at<double>(i, 0) = rng.uniform(-init_val, init_val);
        for (int j = 0; j < 2 * rnn_size; ++j)
            W_.at<double>(i, j) = rng.uniform(-init_val, init_val);
    }
}

void NeuralNetwork::train(const size_t &batch_size, const size_t &train_length,
                          const size_t &valid_size) {
#ifdef USE_OMP
    omp_set_num_threads(12);
#endif
    std::cout << std::endl;
    std::cout << "Starting training process..." << std::endl;
    std::cout << "\tBatch size = " << batch_size << std::endl;
    std::cout << "\tTraining length = " << train_length << std::endl;
    std::cout << std::endl;

    do {
        Generator generator(strategy_);
        size_t idx = 0;
        size_t batch_idx = 0;
        std::vector<std::vector<cv::Mat> > x(batch_size);
        std::vector<std::vector<cv::Mat> > y(batch_size);
        std::vector<cv::Mat> x_valid;
        std::vector<cv::Mat> y_valid;
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
            x[batch_idx].push_back(cv::Mat(sample.input.size(), 1, CV_64F));
            y[batch_idx].push_back(cv::Mat(sample.output.size(), 1, CV_64F));
            for (size_t j = 0; j < sample.input.size(); ++j, ++idx)
                x[batch_idx].back().at<double>(j, 0) = sample.input[j];
            for (size_t j = 0; j < sample.output.size(); ++j)
                y[batch_idx].back().at<double>(j, 0) = sample.output[j];
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
        idx = 0;
        while (true) {
            TrainSample sample = generator.generate();
            if (idx + sample.input.size() >= train_length)
                break;
            x_valid.push_back(cv::Mat(sample.input.size(), 1, CV_64F));
            y_valid.push_back(cv::Mat(sample.output.size(), 1, CV_64F));
            for (size_t j = 0; j < sample.input.size(); ++j, ++idx)
                x_valid.back().at<double>(j, 0) = sample.input[j];
            for (size_t j = 0; j < sample.output.size(); ++j)
                y_valid.back().at<double>(j, 0) = sample.output[j];
        }
        std::cout << "Generation done." << std::endl << std::endl;
        size_t ticks = 0;
        std::cout << "Maximizing aposteriori probability..." << std::endl;
        double learning_rate = 1.0;
        double prev_val = estimateDatasetLikelihood(x_valid, y_valid, NULL);
        std::cout << "Objective value = " << prev_val << std::endl;
        long int srand_param = rand();
        srand(time(&srand_param));
        while (learning_rate > 0.001) {
            if (ticks > 20) {
                std::cout << "Decreasing learning rate" << std::endl;
                learning_rate *= 0.8;
            }
            int cur_batch = rand() % batch_size;
            const std::vector<cv::Mat>& temp_x = x[cur_batch];
            const std::vector<cv::Mat>& temp_y = y[cur_batch];
            BackwardPropogationModule bpm;

            estimateDatasetLikelihood(temp_x, temp_y, &bpm);
            cv::Mat upd_h2o = h2o_ - learning_rate * bpm.grad_h2o;
/*            cv::Mat upd_h2o = h2o - learning_rate * h2o_grad;
            cv::Mat upd_W_encoder = W_encoder - learning_rate * W_encoder_grad;
            cv::Mat upd_W_decoder = W_decoder - learning_rate * W_decoder_grad;
            cv::Mat upd_b_encoder = b_encoder - learning_rate * b_encoder_grad;
            cv::Mat upd_b_decoder = b_decoder - learning_rate * b_decoder_grad;
            cv::swap(upd_i2h, i2h);
            cv::swap(upd_h2o, h2o);
            cv::swap(upd_W_encoder, W_encoder);
            cv::swap(upd_W_decoder, W_decoder);
            cv::swap(upd_b_encoder, b_encoder);
            cv::swap(upd_b_decoder, b_decoder);*/
            cv::swap(h2o_, upd_h2o);
            double upd_val = estimateDatasetLikelihood(x_valid, y_valid);
            if (upd_val < prev_val) {
                prev_val = upd_val;
                ticks = 0;
                std::cout << "Minimal objective val updated = " << upd_val << std::endl;
            } else {
                std::cout << "No progress" << std::endl;
                ++ticks;
                cv::swap(upd_h2o, h2o_);
//                cv::swap(upd_W_encoder, W_encoder);
//                cv::swap(upd_W_decoder, W_decoder);
//                cv::swap(upd_b_encoder, b_encoder);
//                cv::swap(upd_b_decoder, b_decoder);
            }
        }
        std::cout << "Aposteriori probablity maximized." << std::endl;
    } while (strategy_->makeHarder());
    std::cout << "...End of training process" << std::endl;
}


double NeuralNetwork::estimateLikelihood(const cv::Mat &x, const cv::Mat &y,
                                         const cv::Mat &W, cv::Mat &b,
                                         const cv::Mat &input_gate1,
                                         const cv::Mat &input_gate2,
                                         const cv::Mat &h2o,
                                         BackwardPropogationModule* bpm) {
    if (bpm != NULL) {
        bpm->grad_h2o = cv::Mat::zeros(h2o.rows, h2o.cols, CV_64F);
    }
    std::vector<cv::Mat> cur_h(layers_amount_ + 1,
                               cv::Mat::zeros(rnn_size_, 1, CV_64F));
    std::vector<cv::Mat> prev_h(layers_amount_ + 1,
                               cv::Mat::zeros(rnn_size_, 1, CV_64F));
    std::vector<cv::Mat> cur_c(layers_amount_ + 1,
                               cv::Mat::zeros(rnn_size_, 1, CV_64F));
    std::vector<cv::Mat> prev_c(layers_amount_ + 1,
                               cv::Mat::zeros(rnn_size_, 1, CV_64F));
    cv::Mat input_gate = input_gate1;
    bool need_to_output = false;
    int output_idx = 0;
    double res = 0.0;
    SymbolManager* symb_manager = SymbolManager::getInstance();
    for (int i = 0; i < x.rows - 1; ++i) {
        int col_inp;
        if (need_to_output)
            col_inp = symb_manager->getIndexOfOutput(x.at<double>(i, 0));
        else
            col_inp = x.at<double>(i, 0);
        cur_h[0] = input_gate.col(x.at<double>(i, 0) - 1).clone();
        for (size_t j = 1; j < layers_amount_ + 1; ++j) {
            cv::Mat arg(2 * rnn_size_, 1, CV_64F);
            for (int h = 0; h < rnn_size_; ++h)
                arg.at<double>(h, 0) = cur_h[j - 1].at<double>(h, 0);
            for (int h = 0; h < rnn_size_; ++h)
                arg.at<double>(h + rnn_size_, 0) = prev_h[j].at<double>(h, 0);
            cv::Mat out = W * arg + b;
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
        if (std::abs(x.at<double>(i, 0) -
                     symb_manager->getIndexOfSymbol('@')) < 1E-6) {
            need_to_output = true;
            input_gate = input_gate2;
        }
        if (need_to_output) {
            cv::Mat output = h2o * cur_h[cur_h.size() - 1];
            if (bpm != NULL) {
                double output_denom = 0.0;
                for (int j = 0; j < output.rows; ++j)
                    output_denom += exp(output.at<double>(j, 0));
                double h2o_output_nom = exp(output.at<double>(symb_manager->getIndexOfOutput(
                                                    y.at<double>(output_idx, 0)),0));
                for (int j = 0; j < rnn_size_; ++j) {
                    bpm->grad_h2o.at<double>(symb_manager->getIndexOfOutput(
                                                 y.at<double>(output_idx, 0)), j) +=
                            cur_h[cur_h.size() - 1].at<double>(j, 0) * h2o_output_nom / output_denom -
                                cur_h[cur_h.size() - 1].at<double>(j, 0);
                }
            }
            vectorToDistribution(output);
            double prob = output.at<double>(symb_manager->getIndexOfOutput(
                                                y.at<double>(output_idx, 0)), 0);
            res -= log(prob);
            output_idx++;
        }
    }
    return res;
}

double NeuralNetwork::estimateDatasetLikelihood(const std::vector<cv::Mat> &x,
                                                const std::vector<cv::Mat> &y,
                                                BackwardPropogationModule* bpm) {
    double res = 0.0;
    if (bpm != NULL) {
        bpm->grad_h2o = cv::Mat::zeros(h2o_.rows, h2o_.cols, CV_64F);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        if (bpm != NULL ) {
            BackwardPropogationModule local_bpm;
            res += estimateLikelihood(x[i], y[i], W_, b_, input_gate1_,
                                  input_gate2_, h2o_, &local_bpm);
            bpm->grad_h2o += local_bpm.grad_h2o;
        } else {
            res += estimateLikelihood(x[i], y[i], W_, b_, input_gate1_,
                                  input_gate2_, h2o_);
        }

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
