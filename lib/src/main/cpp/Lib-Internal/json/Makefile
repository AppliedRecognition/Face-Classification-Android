INCLUDES	=
DEP_INTERNAL	= applog
DEP_EXTERNAL	= z

LIBS		= json.a
SRCS_json	= types.cpp io_manip.cpp encode.cpp io.cpp \
		  push_types.cpp pull_types.cpp \
		  pull_encode_json.cpp push_decode_json.cpp \
		  pull_encode_amf3.cpp push_decode_amf3.cpp \
		  pull_encode_cbor.cpp push_decode_cbor.cpp \
		  cbor_decode.cpp zlib.cpp

PROGS		= json-inspect

all:	$(LIBS) $(PROGS) test_lib

include ../Makefile.inc
