INCLUDES        =
DEP_INTERNAL    = models raw_image core json
DEP_EXTERNAL    =
PROGS_INTERNAL	= raw_image_io

LIBS		= det.a
SRCS_det	= landmarks.cpp pose.cpp \
		  types.cpp detection_settings.cpp \
		  image.cpp detection.cpp drawing.cpp

all:	$(LIBS) $(EXCLUDED) test_lib

include ../Makefile.inc
