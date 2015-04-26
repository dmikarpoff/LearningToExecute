#ifndef INCLUDE_ILEARNINGSTRATEGY_H_
#define INCLUDE_ILEARNINGSTRATEGY_H_

#include <cstdlib>

struct Hardness {
    Hardness(const size_t& l = 0, const size_t& n = 0):
        length(l), nesting(n) {}
    size_t length;
    size_t nesting;
};

enum LearningStrategyType {
    BASELINE = 0,
    LST_SIZE
};

class ILearningStrategy {
 public:
    virtual Hardness getHardness() const = 0;
    virtual bool makeHarder() = 0;
    virtual ~ILearningStrategy() {}
};

#endif  // INCLUDE_ILEARNINGSTRATEGY_H_
