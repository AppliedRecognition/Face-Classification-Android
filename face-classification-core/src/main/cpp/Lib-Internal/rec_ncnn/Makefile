INCLUDES	=
DEP_INTERNAL	= models rec raw_image
DEP_EXTERNAL    = ncnn gomp

LIBS		= rec_ncnn.a
SRCS_rec_ncnn	= models.cpp extract.cpp engine.cpp

all:	$(LIBS) $(EXCLUDED) test_lib

include ../Makefile.inc
