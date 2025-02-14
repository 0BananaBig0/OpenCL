#################################################################
#											 Makefile
#		Makefile to build exe
#################################################################
.PHONY: clean, all

CC = g++
SHARED_LIB_PATH =

# comment -ltclreadline, use rlwrap instead
SHARED_LIBS =  -lOpenCL

LINK_OPTION = ${SHARED_LIB_PATH} ${SHARED_LIBS}
INCLUDES =

DEBUG = -g

CC_FLAGS = ${DEBUG} -fPIC -Wall -Wextra -std=c++17 \
					-Wno-error=deprecated-declarations

# list of object files
OBJS =

STATIC_LIB =

# list of OpenCL source files
OCL =

# list of header files
HEADERS =

# List of all source files except Fun.cpp, FunN.cpp
SRCS := $(wildcard *.cpp)
EXES := $(patsubst %.cpp,%.exe,$(SRCS))

# Rule to build all executables
all: $(EXES)

# Rule to compile other .cpp files directly into executables
$(EXES): %.exe: %.cpp
	$(CC) $< $(CC_FLAGS) $(LINK_OPTION) $(INCLUDES) -o $@

clean:
	@echo "Cleaning up ......"
	@-rm -rf *.exe *.o *.i *.s *.a *.so # - prefix for ignoring errors
