#
#  Make file for cuda_image_filering_global
#

#
# Macros
#
IMG_LDFLAG	= -ltiff -lpng -ljpeg
LDFLAGS 	= $(IMG_LDFLAG) -lm

CUDA_INCFLAG	= -I/home/ichimura/NVIDIA_CUDA-9.2_Samples/common/inc
INCFLAGS	= $(CUDA_INCFLAG)

CC		= nvcc
CFLAGS		= -gencode arch=compute_30,code=sm_30 \
		  -gencode arch=compute_52,code=sm_52 \
		  -gencode arch=compute_61,code=sm_61 \
		  --fmad=false \
		  -O3 -std=c++11

CPP_SRCS	= path_handler.cpp \
		  padding.cpp \
		  image_rw_cuda.cpp \
		  postprocessing.cpp \
		  get_micro_second.cpp

CPP_HDRS	= path_handler.h \
		  padding.h \
		  image_rw_cuda.h \
		  image_rw_cuda_fwd.h \
		  postprocessing.h \
		  get_micro_second.h

CU_SRCS		= cuda_image_filtering_global.cu

CU_HDRS		= 

CPP_OBJS	= $(CPP_SRCS:.cpp=.o) 
CU_OBJS		= $(CU_SRCS:.cu=.o)
TARGET		= cuda_image_filtering_global

CPP_DEPS	= $(CPP_SRCS:.cpp=.d)
CU_DEPS		= $(CU_SRCS:.cu=.d)
DEP_FILE	= Makefile.dep

#
# Suffix rules
#
.SUFFIXES: .cpp
.cpp.o:
	$(CC) $(INCFLAGS) $(CFLAGS)  -c $<

.SUFFIXES: .cu
.cu.o:
	$(CC) $(INCFLAGS) $(CFLAGS)  -c $<

.SUFFIXES: .d
.cpp.d:
	$(CC) $(INCFLAGS) -M $< > $*.d
.cu.d:
	$(CC) $(INCFLAGS) -M $< > $*.d

#
# Generating the target
#
all: $(DEP_FILE) $(TARGET) 

#
# Linking the execution file
#
$(TARGET) : $(CU_OBJS) $(CPP_OBJS) 
	$(CC) -o $@ $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS)

#
# Generating and including dependencies
#
depend: $(DEP_FILE)
$(DEP_FILE) : $(CPP_DEPS) $(CU_DEPS)
	cat $(CPP_DEPS) $(CU_DEPS) > $(DEP_FILE)
ifeq ($(wildcard $(DEP_FILE)),$(DEP_FILE))
include $(DEP_FILE)
endif

#
# Cleaning the files
#
clean:
	rm -f $(CU_OBJS) $(CPP_OBJS) $(CPP_DEPS) $(CU_DEPS) $(DEP_FILE) $(TARGET) *~
