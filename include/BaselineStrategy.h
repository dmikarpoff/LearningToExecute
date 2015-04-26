#ifndef INCLUDE_BASELINESTRATEGY_H_
#define INCLUDE_BASELINESTRATEGY_H_

#include "ILearningStrategy.h"

class BaselineStrategy : public ILearningStrategy {
 public:
    BaselineStrategy(const size_t& length, const size_t& nesting):
        target_length_(length), target_nesting_(nesting) {}
    Hardness getHardness() const {
        Hardness h;
        h.length = target_length_;
        h.nesting = target_nesting_;
        return h;
    }
    bool makeHarder() {
        return false;
    }

 private:
    size_t target_length_, target_nesting_;
};

#endif  // INCLUDE_BASELINESTRATEGY_H_
