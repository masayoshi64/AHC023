/usr/bin/g++ -O0 -g -D=LOCAL -Wshadow -Wall -fsanitize=address -fsanitize=undefined -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -std=c++17 ../main.cpp -o ../a.out --debug
wait
psytester r results --tests 0-9