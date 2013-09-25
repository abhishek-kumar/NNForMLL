BINDIR=bin
DEPDIR=.deps
LIBLBFGS_PATH=/usr/local/lib/liblbfgs.la

all:
	-rm -rf $(BINDIR)
	-mkdir $(BINDIR)
	mkdir -p $(DEPDIR)
	$(MAKE) linker
	-rm *.o

linker: run.o
	./lib/liblbfgs-1.10/libtool --mode=link --tag=CC g++ -msse2 -DUSE_SSE -O3 -ffast-math  -Wall -msse2 -DUSE_SSE -O3 -ffast-math  -Wall -o bin/mll *.o $(LIBLBFGS_PATH) -lm

run.o: lib
	g++ -DHAVE_CONFIG_H -I. -msse2 -DUSE_SSE -O3 -ffast-math  -Wall -msse2 -DUSE_SSE -O3 -ffast-math  -Wall -MT run.o -MD -MP -MF ".deps/sample.Tpo" -g -c src/run.c src/singleLayerNN.c src/nn.c src/parameters.c src/BRSingleLayerNN.c

lib: 
	@echo "lib/liblbfgs was not found in this directory. Going to attempt install."
	@echo "Press any key to continue, or Ctrl-C to abort"
	@echo
	@read input
	chmod a+x install.sh
	./install.sh

clean:
	-rm -rf $(BINDIR)
	-rm -rf $(DEPDIR)
	sudo rm -rf ./lib

