#ifndef INCLUDE_GENERATOR_H_
#define INCLUDE_GENERATOR_H_

#include <string>
#include <utility>

struct TrainSample {
    std::string source_code;
    std::string result;
};

class Generator {
 public:
    Generator(int nesting, int length):
        nesting_(nesting), length_(length) {}
    void setHardness(int nesting, int length);
    TrainSample generate() const;

 private:
    int nesting_, length_;
};

#endif  // INCLUDE_GENERATOR_H_
