INCLUDES        =
DEP_INTERNAL    = raw_image core
DEP_EXTERNAL    = opencv_calib3d opencv_imgproc opencv_core

LIBS		= render.a
SRCS_render	= opencv_mesh.cpp opencv_fill.cpp \
		  dlib_landmarks.cpp dlib.cpp \
		  frontalize_align.cpp frontalize_render.cpp render.cpp

# objects not included in library but built to ensure they compile
EXCLUDED =

all:	$(LIBS) $(EXCLUDED) test_lib

include ../Makefile.inc
