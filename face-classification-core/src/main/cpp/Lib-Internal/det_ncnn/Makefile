INCLUDES        =
DEP_INTERNAL    = det
DEP_EXTERNAL    = ncnn gomp
PROGS_INTERNAL	= raw_image_io

LIBS		= det_ncnn.a
SRCS_det_ncnn	= v6_rfb320.cpp v7_retinaface.cpp \
		  facemesh.cpp \
		  ncnn_common.cpp init.cpp

all:	$(LIBS) $(EXCLUDED) test_lib

include ../Makefile.inc
