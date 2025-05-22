INCLUDES	=
DEP_INTERNAL	= applog
DEP_EXTERNAL	=
PROGS_INTERNAL	=

LIBS		= raw_image.a
SRCS_raw_image	= reader.cpp reader_scale.cpp reader_rotate.cpp \
		  core.cpp transform.cpp image_quality.cpp \
		  face_types.cpp face_landmarks.cpp \
		  linear_regression.cpp scaled_chip.cpp \
		  input_extractor.cpp input_extractor_retina.cpp \
		  adjust.cpp drawing.cpp

all:	$(LIBS) test_lib

include ../Makefile.inc
