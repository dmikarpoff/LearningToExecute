#include <cstring>

#include "../include/ConfigurationHandler.h"

bool ConfigurationHandler::load(const std::string &path,
                                const std::string &conf_name) {
    pugi::xml_document doc;
    if (!doc.load_file(path.c_str()))
        return false;
    pugi::xml_node root = doc.child("configurations");
    if (root.empty())
        return false;
    for (pugi::xml_node_iterator i = root.begin(); i != root.end(); ++i) {
        if (strcmp(i->name(), "configuration") == 0 &&
                    strcmp(i->attribute("name").as_string(),
                                        conf_name.c_str()) == 0)
            return parseSingleNode(*i);
    }
    return false;
}

bool ConfigurationHandler::parseSingleNode(const pugi::xml_node &node) {
    pugi::xml_node rtype = node.child("task");
    if (rtype.empty())
        return false;
    pugi::xml_attribute task = rtype.attribute("val");
    if (strcmp(task.as_string(), "train") == 0)
        run_type_ = TRAINING;
    else if (strcmp(task.as_string(), "test") == 0)
        run_type_ = TESTING;
    else
        return false;
    pugi::xml_node rnn_size = node.child("rnn_size");
    if (rnn_size.empty())
        return false;
    pugi::xml_attribute rnn_size_attr = rnn_size.attribute("val");
    if (rnn_size_attr.empty())
        return false;
    rnn_size_ = rnn_size_attr.as_uint();
    pugi::xml_node layers_amount = node.child("layers_amount");
    if (layers_amount.empty())
        return false;
    pugi::xml_attribute layers_amount_attr = layers_amount.attribute("val");
    if (layers_amount_attr.empty())
        return false;
    layers_amount_ = layers_amount_attr.as_uint();
    pugi::xml_node init_range = node.child("init_range");
    if (init_range.empty())
        return false;
    pugi::xml_attribute init_range_attr = init_range.attribute("val");
    if (init_range_attr.empty())
        return false;
    init_range_ = init_range_attr.as_double();
    pugi::xml_node target_length = node.child("target_length");
    if (target_length.empty())
        return false;
    pugi::xml_attribute target_length_attr = target_length.attribute("val");
    if (target_length_attr.empty())
        return false;
    target_length_ = target_length_attr.as_uint();
    pugi::xml_node target_nesting = node.child("target_nesting");
    if (target_nesting.empty())
        return false;
    pugi::xml_attribute target_nesting_attr = target_nesting.attribute("val");
    if (target_nesting_attr.empty())
        return false;
    target_nesting_ = target_nesting_attr.as_uint();
    pugi::xml_node strategy = node.child("strategy");
    if (strategy.empty())
        return false;
    pugi::xml_attribute strategy_attr = strategy.attribute("val");
    if (strategy_attr.empty())
        return false;
    if (strcmp(strategy_attr.as_string(), "baseline") == 0)
        strategy_ = BASELINE;
    else
        strategy_ = LST_SIZE;
    pugi::xml_node batch_size = node.child("batch_size");
    if (batch_size.empty())
        return false;
    pugi::xml_attribute batch_size_attr = batch_size.attribute("val");
    if (batch_size_attr.empty())
        return false;
    batch_size_ = batch_size_attr.as_uint();
    pugi::xml_node train_length = node.child("train_length");
    if (train_length.empty())
        return false;
    pugi::xml_attribute train_length_attr = train_length.attribute("val");
    if (train_length_attr.empty())
        return false;
    train_length_ = train_length_attr.as_uint();
    pugi::xml_node valid_length = node.child("valid_length");
    if (valid_length.empty())
        return false;
    pugi::xml_attribute valid_length_attr = valid_length.attribute("val");
    if (valid_length_attr.empty())
        return false;
    valid_length_ = valid_length_attr.as_uint();
    return true;
}
