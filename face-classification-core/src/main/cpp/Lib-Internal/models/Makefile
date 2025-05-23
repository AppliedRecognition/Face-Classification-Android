INCLUDES        =
DEP_INTERNAL    = applog
DEP_EXTERNAL    = boost_filesystem stdc++fs

# comment out these lines to disable specific filesystem implementation
# if disabled, the corresponding DEP_EXTERNAL above can also be removed
MODELS_ENABLE_BOOST = true	# libboost_filesystem
MODELS_ENABLE_STDFS = true	# C++17 filesystem

LIBS		= models.a
SRCS_models	=

ifdef MODELS_ENABLE_BOOST
  SRCS_models += loader_boost.cpp
endif

ifdef MODELS_ENABLE_STDFS
  SRCS_models += loader_std.cpp
endif


all:	$(LIBS) test_lib

include ../Makefile.inc
