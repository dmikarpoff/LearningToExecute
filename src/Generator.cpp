#include "../include/Generator.h"

#include <cstdlib>

void Generator::setHardness(int nesting, int length) {
    nesting_ = nesting;
    length = length_;
    srand(time(NULL));
}

TrainSample Generator::generate() const {
    std::string code = "";
    std::string output = "";
    TrainSample res;
    res.source_code = code;
    res.result = output;
    return res;
}
