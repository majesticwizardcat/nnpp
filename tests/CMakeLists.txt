add_compile_options(-Wall -Wextra -Wpedantic -Werror)

add_executable(MainTest main-test.cpp)
add_executable(SecondMainTest second-main-test.cpp)
add_executable(NNTest nn-test.cpp)
add_executable(NNAiTest nnai-test.cpp)
add_executable(PercTest perc-test.cpp)
add_executable(TrainerTest trainer-test.cpp)
add_executable(PerfTest perf-test.cpp)
