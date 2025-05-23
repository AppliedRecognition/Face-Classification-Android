INCLUDES        =
DEP_INTERNAL    = raw_image core json
DEP_EXTERNAL    = dlib lapack openblas blas
PROGS_INTERNAL	= raw_image_io
PROGS_EXTERNAL	= stdc++fs

# comment out the following line to use blas instead of openblas
# if disabled, then "openblas" may be removed from DEP_EXTERNAL above
DLIBX_ENABLE_OPENBLAS = true

# The files matrix_ops_*.cpp are mutually exclusive -- almost.
# See matrix_ops.hpp for details.

LIBS		= dlibx.a
SRCS_dlibx	= shape_predictor.cpp shape_extract_pixels.cpp shape_quality_lbp.cpp \
		  raw_image.cpp hash32.cpp \
		  input_extractor.cpp input_extractor_facechip.cpp \
		  net_layer.cpp net_layer_impl.cpp net_vector.cpp net_convert.cpp \
		  bfloat16.cpp chip_details.cpp linear_regression.cpp \
		  library_init.cpp \
		  qmat.cpp conv_blas.cpp tensor_conv.cpp \
		  dnn_loss_metric.cpp dnn_loss_cosine.cpp

PROGS		= nv-editor nv-inspect nv-profile

ifdef DLIBX_ENABLE_OPENBLAS
  SRCS_dlibx += blas_openblas.cpp
  # objects not included in library but built to ensure they compile
  EXCLUDED = blas_other.o	# conv_ncnn.o
else
  SRCS_dlibx += blas_other.cpp
endif

# select matrix_ops implementation based on this machine
MACHINE := $(shell uname -m)
ifeq ($(MACHINE),x86_64)
  SRCS_dlibx  += matrix_ops_x86_sse.cpp matrix_ops_x86_avx.cpp matrix_ops_x86_detect.cpp
  EXCLUDED += matrix_ops_x86_sse_only.o matrix_ops_generic.o
  FLAGS_matrix_ops_x86_avx.cpp = -mavx2
else ifeq ($(MACHINE),armv7l)
  SRCS_dlibx  += matrix_ops_armv7l.cpp
else ifeq ($(MACHINE),aarch64)
  SRCS_dlibx  += matrix_ops_aarch64.cpp
endif

all:	$(LIBS) $(PROGS) $(EXCLUDED) test_lib

include ../Makefile.inc
