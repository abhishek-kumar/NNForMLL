mkdir -p bin
rm bin/*

g++ -DHAVE_CONFIG_H -I. -msse2 -DUSE_SSE -O3 -ffast-math  -Wall -msse2 -DUSE_SSE -O3 -ffast-math  -Wall -MT bin/run.o -MD -MP -MF ".deps/sample.Tpo" -g -c run.c singleLayerNN.c nn.c parameters.c BRSingleLayerNN.c

../liblbfgs-1.10/libtool --mode=link --tag=CC g++ -msse2 -DUSE_SSE -O3 -ffast-math  -Wall -msse2 -DUSE_SSE -O3 -ffast-math  -Wall   -o bin/mll *.o /usr/local/lib/liblbfgs.la -lm

#scp *.c *.h *.sh abhishek@hg.ucsd.edu:/home/abhishek/NN/NeuralNetsForMLL/

rm *.o

