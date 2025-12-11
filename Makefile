.PHONY: build test clean

all: build

build:
	mkdir -p build
	cd build && cmake .. && make -j

test:
	mkdir -p build
	cd build && cmake .. && make -j && make test

clean:
	rm -rf build