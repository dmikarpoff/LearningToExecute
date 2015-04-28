#ifndef INCLUDE_GENERATOR_H_
#define INCLUDE_GENERATOR_H_

#include <stack>
#include <string>
#include <map>
#include <utility>
#include <vector>

#include "ILearningStrategy.h"
#include "VariableManager.h"

struct TrainSample {
    std::vector<size_t> input;
    std::vector<size_t> output;
    std::string source_code;
};

class Generator {
 public:
    Generator(ILearningStrategy* strategy);
    TrainSample generate();
    size_t getDictionarySize() {
        return dict_.size();
    }

 private:
    ILearningStrategy* strategy_;
    std::stack<std::pair<std::string, long long> > gstack_;
    VariableManager var_manager_;
    std::map<char, int> dict_;

    struct OperationOutput {
        std::vector<std::string> code;
        std::string expr;
        long long eval;
    };

    typedef OperationOutput (Generator::*func_generator)();

    OperationOutput pair_opr();
    OperationOutput smallmul_opr();
    OperationOutput equality_opr();
    OperationOutput vars_opr();
    OperationOutput small_loop_opr();
    OperationOutput ifstat_opr();
    std::pair<std::string, long long> getOperand();
    std::vector<std::pair<std::string, long long> >
                                getOperand(const size_t& num);
    std::vector<func_generator> random_op;
};

#endif  // INCLUDE_GENERATOR_H_
