SHELL := /bin/bash

CPPFLAGS = -pthread -fopenmp --std=c++11
PLOTTERDIR = "plot"

install: a.out
	cd $(PLOTTERDIR) && make install

a.out:
	g++ $(CPPFLAGS) *.cc

clean:
	@rm -f a.out
	cd $(PLOTTERDIR) && make clean
