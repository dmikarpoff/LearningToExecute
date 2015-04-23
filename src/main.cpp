#include <iostream>

#include "../include/ConfigurationHandler.h"
#include "../include/Network.h"

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cout << "Too much arguments" << std::endl;
        return 0;
    }
    std::string conf_name;
    if (argc == 2) {
        conf_name = argv[1];
        std::cout << "Using configuration " << conf_name << std::endl;
    } else {
        conf_name = "default";
        std::cout << "Using default configuration" << std::endl;
    }

    ConfigurationHandler config;
    if (!config.load("../config.xml", conf_name)) {
        std::cout << "Failed to load configuration" << std::endl;
        return 0;
    }
    if (config.getRunType() == TRAINING) {
        std::cout << "Start training task" << std::endl;
        std::cout << "Initializing network" << std::endl
                  << "\tLayer's size = " << config.getRNNSize() << std::endl
                  << "\tLayer's amount = " << config.getLayersAmount() << std::endl
                  << "\tInit range = " << config.getInitRange() << std::endl;

        NeuralNetwork nn(config.getRNNSize(), config.getLayersAmount(),
                         config.getInitRange());
        std::cout << "End of training task" << std::endl;
    }
    return 0;
}
