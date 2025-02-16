#################################################################
#											 Makefile
#		Makefile to build exe
#################################################################
.PHONY: clean all dbg

CC = g++
SHARED_LIB_PATH =

# comment -ltclreadline, use rlwrap instead
SHARED_LIBS =  -lOpenCL -lpthread -lX11

LINK_OPTION = ${SHARED_LIB_PATH} ${SHARED_LIBS}
INCLUDES = -I. -Istb

DEBUG =

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

dbg:
	$(MAKE) all DBG="-g -DDBG -O0"

# Rule to build all executables
all: $(EXES)

# Rule to compile other .cpp files directly into executables
$(EXES): %.exe: %.cpp
	$(CC) $(DBG) $< $(CC_FLAGS) $(LINK_OPTION) $(INCLUDES) -o $@

clean:
	@echo "Cleaning up ......"
	@-rm -rf *.exe *.o *.i *.s *.a *.so # - prefix for ignoring errors
