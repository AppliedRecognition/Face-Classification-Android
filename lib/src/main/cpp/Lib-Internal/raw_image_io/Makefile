INCLUDES	=
DEP_INTERNAL	= raw_image
DEP_EXTERNAL	= png tiff jpeg
PROGS_INTERNAL	= json
#PROGS_EXTERNAL	= exif

#
# Note: tiff and png support are optional.
#
# To include support include the corresponding _enable.cpp file
# in your projet.  Also link to the necessary external libraries.
#
# To exclude support you must include the _disable.cpp file instead.
# An exception will be thrown on attempt to load such images.
#

LIBS = raw_image_io.a
SRCS_raw_image_io = \
	png_enable.cpp tiff_enable.cpp \
	jpeg_load.cpp jpeg_save.cpp io.cpp

# objects not included in library but built to ensure they compile
EXCLUDED = png_disable.o tiff_disable.o

all:	$(LIBS) $(EXCLUDED) test_lib

include ../Makefile.inc
