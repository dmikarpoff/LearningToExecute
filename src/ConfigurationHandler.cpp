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
    pugi::xml_attribute rnn_size_attr = rtype.attribute("val");
    if (rnn_size_attr.empty())
        return false;
    rnn_size_ = rnn_size_attr.as_uint();
    pugi::xml_node layers_amount = node.child("layers_amount");
    if (layers_amount.empty())
        return false;
    pugi::xml_attribute layers_amount_attr = rtype.attribute("val");
    if (layers_amount_attr.empty())
        return false;
    layers_amount_ = layers_amount_attr.as_uint();
    return true;
}
