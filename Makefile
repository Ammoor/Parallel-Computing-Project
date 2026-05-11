# =====================================================================
# Makefile - PC 2026 Project
# =====================================================================
CXX       = mpicxx
CXXFLAGS  = -O2 -std=c++14 -Wall
TARGET    = pc_project
SRCDIR    = src
SOURCES   = $(SRCDIR)/main.cpp \
            $(SRCDIR)/heat_diffusion.cpp \
            $(SRCDIR)/prefix_sum.cpp \
            $(SRCDIR)/deadlock_demo.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) data/heat_output.txt data/prefix_output.txt

.PHONY: all clean
