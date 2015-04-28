#include "../include/Generator.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

Generator::Generator(ILearningStrategy *strategy):
        strategy_(strategy) {
    random_op.push_back(&Generator::pair_opr);
    random_op.push_back(&Generator::smallmul_opr);
    random_op.push_back(&Generator::equality_opr);
    random_op.push_back(&Generator::vars_opr);
    random_op.push_back(&Generator::small_loop_opr);
    random_op.push_back(&Generator::ifstat_opr);
    int last_idx = 1;
    for (char c = '0'; c <= '9'; ++c, ++last_idx)
        dict_[c] = last_idx;
    for (char c = 'a'; c <= 'z'; ++c, ++last_idx)
        dict_[c] = last_idx;
    dict_[' '] = last_idx++;
    dict_['#'] = last_idx++;
    dict_['@'] = last_idx++;
    dict_['.'] = last_idx++;
    dict_[':'] = last_idx++;
    dict_['+'] = last_idx++;
    dict_['-'] = last_idx++;
    dict_['*'] = last_idx++;
    dict_['('] = last_idx++;
    dict_[')'] = last_idx++;
    dict_['<'] = last_idx++;
    dict_['>'] = last_idx++;
    dict_['='] = last_idx++;
    dict_[0] = 0;
}

TrainSample Generator::generate() {
    srand(time(NULL));
    while (!gstack_.empty())
        gstack_.pop();
    var_manager_.clear();
    TrainSample res;
    OperationOutput output;
    size_t nesting = strategy_->getHardness().nesting;
    for (size_t i = 0; i < nesting; ++i) {
        int f_idx = rand() % random_op.size();
        OperationOutput single_op = ((*this).*(random_op[f_idx]))();
        for (size_t j = 0; j < single_op.code.size(); ++j)
            output.code.push_back(single_op.code[j]);
        gstack_.push(std::make_pair(single_op.expr, single_op.eval));
    }
    output.eval = gstack_.top().second;
    output.expr = gstack_.top().first;
    std::string target = std::to_string(output.eval) + ".";
    std::string input = "";
    for (size_t i = 0; i < output.code.size(); ++i)
        input += output.code[i] + '#';
    input += "print(" + output.expr + ")@";
    for (size_t i = 0; i < input.size(); ++i) {
        res.output.push_back(0);
        res.input.push_back(dict_.at((unsigned char)input[i]));
    }
    for (size_t i = 0; i < target.size(); ++i) {
        res.output.push_back(dict_.at((unsigned char)target[i]));
        res.input.push_back(0);
    }
    res.source_code = input + target;
    return res;
}

Generator::OperationOutput Generator::pair_opr() {
    std::vector<std::pair<std::string, long long> >
            op = getOperand(2);
    OperationOutput output;
    if (rand() % 2 == 1) {
        output.eval = op[0].second + op[1].second;
        output.expr = "(" + op[0].first + "+" +
            op[1].first + ")";
    } else {
        output.eval = op[0].second - op[1].second;
        output.expr = "(" + op[0].first + "-" +
            op[1].first + ")";
    }
    return output;
}

Generator::OperationOutput Generator::smallmul_opr() {
    std::pair<std::string, long long> operand = getOperand();
    long long b = rand() % (4 * strategy_->getHardness().length);
    OperationOutput output;
    output.eval = b * operand.second;
    if (rand() % 2 == 1) {
        output.expr = "(" + operand.first + "*" +
                std::to_string(b) + ")";
    } else {
        output.expr = "(" + std::to_string(b) + "*" +
                operand.first + ")";
    }
    return output;
}

Generator::OperationOutput Generator::equality_opr() {
    OperationOutput output;
    auto operand = getOperand();
    output.eval = operand.second;
    output.expr = operand.first;
    return output;
}

Generator::OperationOutput Generator::vars_opr() {
    char var = var_manager_.getUnused(1)[0];
    auto operands = getOperand(2);
    OperationOutput output;
    if (rand() % 2 == 1) {
        output.eval = operands[0].second + operands[1].second;
        output.code.push_back(std::string("") + var +
                              ('=' + operands[0].first));
        output.expr = std::string("(") + var + '+' +
                      operands[1].first + ")";
    } else {
        output.eval = operands[0].second - operands[1].second;
        output.code.push_back(std::string("") + var + ('=' + operands[0].first));
        output.expr = std::string("(") + var + '-' +
                      operands[1].first + ")";
    }
    return output;
}

Generator::OperationOutput Generator::small_loop_opr() {
    char var = var_manager_.getUnused(1)[0];
    auto operands = getOperand(2);
    long long loop = rand() % (4 * strategy_->getHardness().length);
    char op;
    OperationOutput output;
    if (rand() % 2 == 1) {
        op = '+';
        output.eval = operands[0].second + loop * operands[1].second;
    } else {
        op = '-';
        output.eval = operands[0].second - loop * operands[1].second;
    }
    output.code.push_back(var + ('=' + operands[0].first));
    output.code.push_back("for x in range(" + std::to_string(loop) + "):" +
                    var + op + '=' + operands[1].first);
    output.expr = var;
    return output;
}

Generator::OperationOutput Generator::ifstat_opr() {
    auto ops = getOperand(4);
    OperationOutput output;
    char op;
    if (rand() % 2 == 1) {
        op = '>';
        if (ops[0].second > ops[1].second)
            output.eval = ops[2].second;
        else
            output.eval = ops[3].second;
    } else {
        op = '<';
        if (ops[0].second < ops[1].second)
            output.eval = ops[2].second;
        else
            output.eval = ops[3].second;
    }
    output.expr = '(' + ops[2].first + " if " + ops[0].first +
            op + ops[1].first + " else " + ops[3].first + ')';
    return output;
}

std::pair<std::string, long long> Generator::getOperand() {
    if (gstack_.empty()) {
        size_t l = strategy_->getHardness().length;
        size_t len = 1;
        for (size_t i = 0; i < l; ++i)
            len *= 10;
        long long eval = rand() % len;
        std::stringstream ss;
        ss << eval;
        return std::make_pair(ss.str(), eval);
    } else {
        std::pair<std::string, long long> res = gstack_.top();
        gstack_.pop();
        return res;
    }
}

int custom_random(int i) {
    return rand() % i;
}

std::vector<std::pair<std::string, long long> > Generator::
                                            getOperand(const size_t &num) {
    std::vector<std::pair<std::string, long long> > res(num);
    std::vector<int> perm(num);
    for (size_t i = 0; i < perm.size(); ++i)
        perm[i] = i;
    std::random_shuffle(perm.begin(), perm.end(), custom_random);
    for (size_t i = 0; i < num; ++i)
        res[perm[i]] = getOperand();
    return res;
}
