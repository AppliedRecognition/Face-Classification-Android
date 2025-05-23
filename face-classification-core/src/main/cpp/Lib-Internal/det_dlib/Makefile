INCLUDES        =
DEP_INTERNAL    = det dlibx
DEP_EXTERNAL    =
PROGS_INTERNAL	= raw_image_io

LIBS		= det_dlib.a
SRCS_det_dlib	= v3_dlib_hog.cpp v4_dlib_cnn.cpp v5_faceapi_tiny.cpp \
		  v6_rfb320.cpp v7_retinaface.cpp \
		  dlib5.cpp dlib68.cpp facemesh.cpp \
		  classifiers.cpp types.cpp init.cpp

all:	$(LIBS) $(EXCLUDED) test_lib

include ../Makefile.inc
