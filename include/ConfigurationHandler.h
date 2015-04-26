#ifndef INCLUDE_CONFIGURATIONHANDLER_H_
#define INCLUDE_CONFIGURATIONHANDLER_H_

#include <pugixml.hpp>

#include <utility>

#include "../include/ILearningStrategy.h"

enum RunType {
    TRAINING,
    TESTING
};

class ConfigurationHandler {
 public:
    bool load(const std::string& path, const std::string &conf_name);
    RunType getRunType() const {
        return run_type_;
    }
    size_t getRNNSize() const {
        return rnn_size_;
    }
    size_t getLayersAmount() const {
        return layers_amount_;
    }
    double getInitRange() const {
        return init_range_;
    }
    size_t getTargetLength() const {
        return target_length_;
    }
    size_t getTargetNesting() const {
        return target_nesting_;
    }
    LearningStrategyType getLearningStrategy() const {
        return strategy_;
    }
    size_t getBatchSize() const {
        return batch_size_;
    }
    size_t getTrainLength() const {
        return train_length_;
    }

 private:
    bool parseSingleNode(const pugi::xml_node& node);

    RunType run_type_;
    size_t rnn_size_;
    size_t layers_amount_;
    size_t target_length_;
    size_t target_nesting_;
    size_t batch_size_;
    size_t train_length_;
    double init_range_;
    LearningStrategyType strategy_;
};

#endif  // INCLUDE_CONFIGURATIONHANDLER_H_
