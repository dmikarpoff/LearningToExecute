#ifndef INCLUDE_CONFIGURATIONHANDLER_H_
#define INCLUDE_CONFIGURATIONHANDLER_H_

#include <pugixml.hpp>

#include <utility>

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

 private:
    bool parseSingleNode(const pugi::xml_node& node);

    RunType run_type_;
    size_t rnn_size_;
    size_t layers_amount_;
    double init_range_;
};

#endif  // INCLUDE_CONFIGURATIONHANDLER_H_
