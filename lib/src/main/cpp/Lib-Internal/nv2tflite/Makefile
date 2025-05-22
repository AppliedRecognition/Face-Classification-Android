INCLUDES	=
DEP_INTERNAL	= dlibx
DEP_EXTERNAL	= tflite ncnn gomp
PROGS_INTERNAL	= raw_image_io

LIBS		= conv.a
SRCS_conv	= conv_tflite.cpp tflite_infer.cpp

PROGS		= nv2tflite tflite2nv

all:	$(LIBS) $(PROGS) # test_lib

include ../Makefile.inc
