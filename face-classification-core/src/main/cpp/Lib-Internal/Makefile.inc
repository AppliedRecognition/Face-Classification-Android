VERSION_OVERRIDE ?=
CC		:= gcc$(VERSION_OVERRIDE)
CXX		:= g++$(VERSION_OVERRIDE)
LINK		:= g++$(VERSION_OVERRIDE)
DOXYGEN		:= doxygen
SHELL		:= /bin/bash
SVN		:= svn

CXX_VERSION	:= $(shell $(CXX) -dumpfullversion -dumpversion |cut -f1,2 -d.)
MACHINE		:= $(shell uname -m)

PROJECT_BASE	?= ../..
INTERNAL_BASE	:= $(PROJECT_BASE)/lib-internal
EXTERNAL_BASE	:= $(PROJECT_BASE)/lib-external


# Within individual Makefile dependencies are defined with:
#
# Dependencies for the LIBS (.a or .so) library files:
#   DEP_INTERNAL =
#   DEP_EXTERNAL =
# These dependencies are recursively determined.
#
# Dependencies for the PROGS and unit tests:
#   PROGS_INTERNAL =
#   PROGS_EXTERNAL =
# old (deprecated) names:
#   LIB_INTERNAL =
#   LIB_EXTERNAL =
# Note: the old (deprecated) names are confusing as they apply to
# the programs being compiled (not the libs)


#### internal dependencies ####

prefix_internal_dir = $(shell for i in $(INTERNAL_BASE)/$(1) $(EXTERNAL_BASE)/*/$(1) $(EXTERNAL_BASE)/$(1) ../$(1); do if [ -f $$i/Makefile ]; then echo $$i; break; fi; done)

read_internal_deps = $(shell egrep '^DEP_INTERNAL' "$(1)" |sed 's/.*= *//; s/  */ /g; s/ $$//')

find_internal_deps = $(foreach d,$(1),$(d) $(call find_internal_deps,$(call read_internal_deps,$(call prefix_internal_dir,$(d))/Makefile)))

# remove duplicates such that the last appearance remains
remove_dups = $(shell for i in $(1); do echo $$i; done |tac |awk '!seen[$$0]++'| tac)

DEP_INTERNAL	:= $(call remove_dups,$(call find_internal_deps,$(DEP_INTERNAL)))
PROGS_INTERNAL	:= $(call remove_dups,$(call find_internal_deps,$(PROGS_INTERNAL) $(LIB_INTERNAL)))
LINK_INTERNAL	:= $(foreach d,$(call remove_dups,$(PROGS_INTERNAL) $(DEP_INTERNAL)),$(call prefix_internal_dir,$(d))/$(d).a)
FLAGS_INTERNAL	:= $(foreach d,$(call remove_dups,$(PROGS_INTERNAL) $(DEP_INTERNAL)),-DLIB_INTERNAL_HAS_$(d))


#### external dependencies ####

read_external_deps = $(shell egrep '^DEP_EXTERNAL' "$(1)" |sed 's/.*= *//; s/  */ /g; s/ $$//')

DEP_EXTERNAL	:= $(call remove_dups,$(DEP_EXTERNAL) $(foreach d,$(PROGS_INTERNAL) $(DEP_INTERNAL),$(call read_external_deps,$(call prefix_internal_dir,$(d))/Makefile)))
PROGS_EXTERNAL	:= $(call remove_dups,$(PROGS_EXTERNAL) $(LIB_EXTERNAL))
LINK_EXTERNAL	:= $(foreach d,$(call remove_dups,$(DEP_EXTERNAL) $(PROGS_EXTERNAL)),-l$(d))


# locate boost libraries
BOOST_LIB_FILE := $(shell find /usr/lib -name 'libboost_*.a' |head -1)
ifneq (,$(BOOST_LIB_FILE))
  BOOST_TAG := $(findstring -mt, $(BOOST_LIB_FILE))
  BOOST_LIB_DIR := $(dir $(BOOST_LIB_FILE))
  ifdef BOOST_STATIC
    LINK_EXTERNAL := $(patsubst -lboost_%,-l:libboost_%$(BOOST_TAG).a,$(LINK_EXTERNAL))
  endif
  BOOST_UNIT_TEST_FRAMEWORK = $(BOOST_LIB_DIR)libboost_unit_test_framework$(BOOST_TAG).a
  #$(info $(BOOST_LIB_DIR))
  #$(info $(BOOST_UNIT_TEST_FRAMEWORK))
endif



# opencv
ifdef OPENCV_STATIC
  LINK_EXTERNAL	:= $(patsubst -lopencv_%,-l:libopencv_%.a,$(LINK_EXTERNAL)) -ldl
endif
ifneq (,$(wildcard /usr/include/opencv4))
  INCLUDES	+= -isystem /usr/include/opencv4
endif


# dlib override to support custom compiled version
DLIB_PREFIX	:= $(EXTERNAL_BASE)/dlib-proper
ifneq (,$(wildcard $(DLIB_PREFIX)/build/dlib/libdlib.a))
  INCLUDES	+= -isystem $(DLIB_PREFIX)/build -isystem $(DLIB_PREFIX)
  LINK_EXTERNAL	:= $(patsubst -ldlib,$(DLIB_PREFIX)/build/dlib/libdlib.a,$(LINK_EXTERNAL))
endif


# tencent ncnn in lib-external
NCNN_PREFIX	:= $(EXTERNAL_BASE)/ncnn
ifneq (,$(wildcard $(NCNN_PREFIX)/build/src/libncnn.a))
  INCLUDES	+= -isystem $(NCNN_PREFIX)/build/src -isystem $(NCNN_PREFIX)/src
  LINK_EXTERNAL	:= $(patsubst -lncnn,$(NCNN_PREFIX)/build/src/libncnn.a,$(LINK_EXTERNAL))
endif


# Tensorflow Lite in lib-external
TFLITE_PREFIX	:= $(EXTERNAL_BASE)/tensorflow
TFLITE_BUILD	:= $(TFLITE_PREFIX)/build
TFLITE_LIB	:= $(TFLITE_BUILD)/libtensorflow-lite.a
ifneq (,$(wildcard $(TFLITE_LIB)))
  TFLITE_DEPS := $(TFLITE_BUILD)/_deps
  ifneq (,$(findstring -ltflite,$(LINK_EXTERNAL)))
    INCLUDES	+= -isystem $(TFLITE_PREFIX)
    INCLUDES	+= -isystem $(TFLITE_BUILD)/flatbuffers/include
    LINK_EXTERNAL := $(patsubst -ltflite,$(TFLITE_LIB),$(LINK_EXTERNAL))
    RUY_DEPS := frontend context prepare_packed_matrices trmul ctx context_get_ctx block_map prepacked_cache apply_multiplier denormal have_built_path_for_avx have_built_path_for_avx2_fma have_built_path_for_avx512 system_aligned_alloc tune thread_pool blocking_counter wait cpuinfo allocator kernel_avx kernel_avx2_fma kernel_avx512 pack_avx pack_avx2_fma pack_avx512
    LINK_EXTERNAL += $(foreach dep,$(RUY_DEPS),$(TFLITE_DEPS)/ruy-build/ruy/libruy_$(dep).a)
    LINK_EXTERNAL += $(foreach dep,fftsg fftsg2d,$(TFLITE_DEPS)/fft2d-build/libfft2d_$(dep).a)
    LINK_EXTERNAL += $(TFLITE_DEPS)/xnnpack-build/libXNNPACK.a
    LINK_EXTERNAL += $(TFLITE_DEPS)/flatbuffers-build/libflatbuffers.a
    LINK_EXTERNAL += $(TFLITE_DEPS)/farmhash-build/libfarmhash.a
    LINK_EXTERNAL += $(TFLITE_DEPS)/cpuinfo-build/libcpuinfo.a
    LINK_EXTERNAL += $(TFLITE_BUILD)/pthreadpool/libpthreadpool.a
  endif
endif


# librealsense for Intel Realsense 3d camera
RS_PREFIX	:= $(shell readlink -e $(EXTERNAL_BASE)/realsense)
ifneq (,$(wildcard $(RS_PREFIX)/librealsense2.so))
  INCLUDES	+= -isystem $(RS_PREFIX)/include
  LINK_EXTERNAL	+= -L$(RS_PREFIX) -Wl,-rpath=$(RS_PREFIX)
endif


# astra sdk for Orbbec 3d camera
ASTRA_PREFIX	:= $(shell readlink -e $(EXTERNAL_BASE)/astra)
ifneq (,$(wildcard $(ASTRA_PREFIX)/lib/libastra.so))
  INCLUDES	+= -isystem $(ASTRA_PREFIX)/include
  LINK_EXTERNAL	+= -L$(ASTRA_PREFIX)/lib -Wl,-rpath=$(ASTRA_PREFIX)/lib
endif


# Vzense SDK for DCAM710 ToF 3d camera
VZENSE_PREFIX	:= $(shell readlink -e $(EXTERNAL_BASE)/vzense)
ifneq (,$(wildcard $(VZENSE_PREFIX)/Lib/libvzense_api.so))
  INCLUDES	+= -isystem $(VZENSE_PREFIX)/Include
  LINK_EXTERNAL	+= -L$(VZENSE_PREFIX)/Lib -Wl,-rpath=$(VZENSE_PREFIX)/Lib
endif


# external libs that can be linked statically if found
ALLOW_STATIC = -lsqlite3 -lcrypto -lssl -ldlib -lexif -lpng -ljpeg -lopenblas -lblas -llapack

ifdef ENABLE_STATIC
  find_static_lib = $(shell find /usr/lib* -xtype f -name '$(1:-l%=lib%).a' |awk '{s=1; print "$(1:-l%=-l:lib%).a"} END {if (!s) print "$(1)"}')
  LINK_EXTERNAL := $(foreach lib,$(LINK_EXTERNAL),$(if $(filter $(lib),$(ALLOW_STATIC)),$(call find_static_lib,$(lib)),$(lib)))
endif


# diagnostic
ifeq ($(DIAG),1)
  $(info $(CURDIR) $@)
  $(info - PROGS_INTERNAL $(PROGS_INTERNAL))
  $(info - DEP_INTERNAL $(DEP_INTERNAL))
  $(info - LINK_INTERNAL $(LINK_INTERNAL))
  $(info - FLAGS_INTERNAL $(FLAGS_INTERNAL))
  $(info - PROGS_EXTERNAL $(PROGS_EXTERNAL))
  $(info - DEP_EXTERNAL $(DEP_EXTERNAL))
  $(info - LINK_EXTERNAL $(LINK_EXTERNAL))
  DIAG := 2
else ifeq ($(DIAG),2)
  $(info $(CURDIR) $@)
  $(info - DEP_INTERNAL $(DEP_INTERNAL))
  $(info - DEP_EXTERNAL $(DEP_EXTERNAL))
endif


#### compiler flags ####

WARNINGS :=	-Wall -Wextra \
		-Wctor-dtor-privacy -Wnon-virtual-dtor \
		-Woverloaded-virtual -Wsign-promo -Wnoexcept \
		-Wcast-qual -Wcast-align \
		-Wredundant-decls -Wstrict-null-sentinel \
		-Wpointer-arith -Wdisabled-optimization \
		-Winvalid-memory-model -Winvalid-pch \
		-Wlogical-op -Wpacked -Wtrampolines \
		-Wredundant-decls -Wstack-protector \
		-Wvariadic-macros -Wvector-operation-performance \
		-Wvirtual-move-assign -Wvla -Woverlength-strings \
		-Wold-style-cast -Wnormalized -Wdate-time \
		-Wconditionally-supported -Wsuggest-override \
		-Wpedantic -Wmissing-declarations \
		-Wconversion -Wsign-conversion -Wfloat-equal \
		-Wattributes \
		-Wcomments -Wconversion-null -Wdelete-incomplete \
		-Wfatal-errors -Wfloat-conversion \
		-Wformat-nonliteral -Wformat-security \
		-Wformat-signedness -Wformat-y2k \
		-Wmissing-include-dirs -Wpacked-bitfield-compat \
		-Wshift-count-negative -Wshift-count-overflow \
		-Wsized-deallocation -Wsizeof-array-argument -Wswitch-bool \
		-Wsync-nand -Wvarargs -Wwrite-strings -Wundef \
		-Wmissing-format-attribute \
		-Wunused -Wunused-but-set-variable -Wunused-local-typedefs \
		-Wunused-macros -Wunused-result \
		-Wframe-address -Whsa -Wignored-attributes -Wplacement-new \
		-Wshift-overflow -Wsubobject-linkage -Wduplicated-cond \
		-Wno-suggest-attribute=format


# additional warnings for newer compilers
WARN_GCC7 :=	-Waligned-new -Walloca -Walloc-zero -Wdangling-else \
		-Wexpansion-to-defined -Wformat-overflow -Wformat-truncation \
		-Wpointer-compare -Wstringop-overflow -Wswitch-unreachable

WARN_GCC8 :=	$(WARN_GCC7) \
		-Wattribute-alias -Wclass-memaccess -Wdelete-non-virtual-dtor \
		-Wextra-semi -Wif-not-aligned -Wliteral-suffix \
		-Wlto-type-mismatch -Wnoexcept-type -Wpacked-not-aligned \
		-Wregister -Wreorder -Wstringop-truncation

WARN_GCC9 :=	$(WARN_GCC8)

WARN_GCC10 :=	$(WARN_GCC9) -Wmismatched-tags -Wredundant-tags

WARN_GCC11 :=	$(WARN_GCC10)

WARN_GCC12 :=	$(WARN_GCC11) \
		-Wno-maybe-uninitialized -Wno-alloc-zero -Wno-array-bounds

WARN_GCC13 :=	$(WARN_GCC12)


ifeq ($(CXX_VERSION:6.%=6.x),6.x)
  # minimum required version
else ifeq ($(CXX_VERSION:7.%=7.x),7.x)
  WARNINGS += $(WARN_GCC7)
else ifeq ($(CXX_VERSION:8.%=8.x),8.x)
  WARNINGS += $(WARN_GCC8)
else ifeq ($(CXX_VERSION:9.%=9.x),9.x)
  WARNINGS += $(WARN_GCC9)
else ifeq ($(CXX_VERSION:10.%=10.x),10.x)
  WARNINGS += $(WARN_GCC10)
else ifeq ($(CXX_VERSION:11.%=11.x),11.x)
  WARNINGS += $(WARN_GCC11)
else ifeq ($(CXX_VERSION:12.%=12.x),12.x)
  WARNINGS += $(WARN_GCC12)
else ifeq ($(CXX_VERSION:13.%=13.x),13.x)
  WARNINGS += $(WARN_GCC13)
else
  $(warning compiler version $(CXX_VERSION) not supported)
  WARNINGS += $(WARN_GCC13)
endif


ifeq (lib-internal,$(findstring lib-internal,$(CURDIR))) 
  # warnings for lib-internal
  WARNINGS += -Werror=old-style-cast

else ifeq (esearch/src/,$(findstring esearch/src/,$(CURDIR)))
  # warnings for research code
  WARNINGS += -Werror=old-style-cast
  WARNINGS += -Wno-conversion -Wno-float-conversion -Wno-sign-conversion
  WARNINGS += -Wno-pedantic
  WARNINGS += -Wno-unused-function
  WARNINGS += -Wno-extra-semi
  WARNINGS += -Wno-noexcept

else # everything else (e.g. services)
  WARNINGS += -Wno-conversion -Wno-sign-conversion
  WARNINGS += -Wno-pedantic
  WARNINGS += -Wno-extra-semi
endif


# Notes regarding gcc warnings.

#  WARNINGS += -Wabi     # one warning about empty classes
#  WARNINGS += -Weffc++  # virtual destructors and member inits

#  WARNINGS += -Wdouble-promotion
#  WARNINGS += -Wunused-const-variable
#  WARNINGS += -Wuseless-cast
#  WARNINGS += -Wswitch-default
#  WARNINGS += -Wswitch-enum  # triggers even with default
#  WARNINGS += -Wsuggest-final-methods -Wsuggest-final-types
#  WARNINGS += -Wmissing-noreturn -Wsuggest-attribute=noreturn

#  WARNINGS += -Winline  # destructors not labelled inline
#  WARNINGS += -Wduplicated-branches  # from opencv
#  WARNINGS += -Wnull-dereference     # from boost::intrusive::set
#  WARNINGS += -Wzero-as-null-pointer-constant   # from boost::variant
#  WARNINGS += -Wunsafe-loop-optimizations  # too many false positives

# complete nonsense warnings for simple things like "if (x > 0)"
#  WARNINGS += -Wstrict-overflow

# [[gnu:pure]] and [[gnu:const]] not available on Windows
#  WARNINGS += -Wsuggest-attribute=pure
#  WARNINGS += -Wsuggest-attribute=const

ifeq ($(DIAG),2)
  $(info - WARNINGS $(WARNINGS))
endif

# consider using -fstack-protector and -fstack-check

CXXREPO		= -fPIC
#CXXREPO	= -frepo
#CXXDEBUG	= -D_GLIBCXX_DEBUG

ifeq ($(MACHINE),x86_64)
  CPUFLAGS	= $(shell grep -o "sse[^ ]*" /proc/cpuinfo |sort |uniq |sed 's/^/-m/; s/_/./')
  #CPUFLAGS	+= $(shell grep -o "avx[^ ]*" /proc/cpuinfo |sort |uniq |sed 's/^/-m/')
else ifeq ($(MACHINE),armv7l)
  CPUFLAGS	= -mfloat-abi=hard -mfpu=neon
else ifeq ($(MACHINE),aarch64)
  CPUFLAGS	=  # neon is mandatory on 64-bit arm
endif

PLATFORM	= -DLINUX -DUNIX_ENV -D_FILE_OFFSET_BITS=64
PREFFLAGS	= $(INCLUDES) -I$(INTERNAL_BASE) $(PLATFORM) $(CPUFLAGS) $(FLAGS_INTERNAL) -O3
CFLAGS		= $(PREFFLAGS) -Wall
BUILD_APPS	= -D__LIB_INTERNAL_BUILD_APPLICATIONS__
APPLOG		= -D__FBLIB_APPLOG_REMOVE_DEPRECATED__ -DAPPLOG_MINIMUM_LEVEL=logTRACE
CXXFLAGS	= $(PREFFLAGS) -std=c++17 $(WARNINGS) $(CXXREPO) $(CXXDEBUG) $(BUILD_APPS) $(APPLOG)
LDFLAGS		= $(CPUFLAGS) $(CXXREPO) -pthread


#### rules ####

.PHONY:	clean headers all

$(LINK_INTERNAL): FORCE
	@$(MAKE) -s -C `dirname $@` `basename $@`
FORCE:

clean:
	@echo "  CLEAN " `basename \`pwd\``
	@rm -rf build	# directory
	@rm -f *% *~ *.[ado] tests/*.[do] *.gch *.hpp-o *.cclog
	@rm -f .empty_file a.out test_lib test_all $(LIBS) $(PROGS)

build/version.h::
	@mkdir -p build
	@if [ -x ./build_version.sh ]; then \
		./build_version.sh >$@.tmp; \
		if ! diff -q $@ $@.tmp >/dev/null 2>/dev/null; then \
			mv -f $@.tmp $@; \
			cat $@; \
		else rm -f $@.tmp; \
		fi \
	fi

build/%.d:	%.c
	@echo "  DEP   " `basename \`pwd\``/$<
	@set -e; mkdir -p build; rm -f $@ $@.*; \
	 $(CC) -MM $(CFLAGS) $< > $@.$$$$; \
	 sed 's,\([^:]*\)\.o[ :]*,build/$*.o $@ : ,g' < $@.$$$$ > $@; \
	 rm -f $@.$$$$

build/%.d:	%.cpp | build/version.h
	@echo "  DEP   " `basename \`pwd\``/$<
	@set -e; mkdir -p build; rm -f $@ $@.*; \
	 $(CXX) -MM $(CXXFLAGS) $< > $@.$$$$; \
	 sed 's,\([^:]*\)\.o[ :]*,build/$*.o $@ : ,g' < $@.$$$$ > $@; \
	 rm -f $@.$$$$

build/tests/%.d:	tests/%.cpp
	@echo "  DEP   " `basename \`pwd\``/$<
	@set -e; mkdir -p build/tests; rm -f $@ $@.*; \
	 $(CXX) -MM $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\([^:]*\)\.o[ :]*,build/tests/$*.o $@ : ,g' < $@.$$$$ > $@; \
	 rm -f $@.$$$$

# the following empty targets ensure the following include doesn't fail
# if a header file no longer exists and is no longer needed, but is
# still present in the .d file
%.h: ;
%.hpp: ;

ifeq ($(filter %clean,$(MAKECMDGOALS)),)
-include $(shell ls *.c *.cpp tests/*.cpp 2>/dev/null |sed 's/[.]cp*$$/.d/; s%^%build/%')
endif

build/%.o:	%.c
	@echo "  C     " `basename \`pwd\``/$<
	@set -o pipefail ;\
	$(CC) $(CFLAGS) -c -o $@ $< 2>&1 \
	  |(if [ -z "$(WRITE_LOGS)" ]; then cat; else tee $<.cclog; fi)

build/%.o:	%.cpp
	@echo "  C++   " `basename \`pwd\``/$<
	@set -o pipefail ;\
	$(CXX) $(CXXFLAGS) $(FLAGS_$<) -c -o $@ $< 2>&1 \
	  |(if [ -z "$(WRITE_LOGS)" ]; then cat; else tee $<.cclog; fi)

%.o:	%.cpp
	@echo "  C++   " `basename \`pwd\``/$<
	@set -o pipefail ;\
	$(CXX) $(CXXFLAGS) $(FLAGS_$<) -c -o $@ $< 2>&1 \
	  |(if [ -z "$(WRITE_LOGS)" ]; then cat; else tee $<.cclog; fi)

%.S:	%.cpp
	$(CXX) $(CXXFLAGS) $(FLAGS_$<) -S -o $@ $<


TESTS_SRC = $(shell ls tests/*.cpp 2>/dev/null)
TESTS_OBJ = $(TESTS_SRC:%.cpp=build/%.o)

tests:: $(TESTS_OBJ)

test_lib: $(INTERNAL_BASE)/applog/unit_tests.o $(TESTS_OBJ) $(LIBS) $(LINK_INTERNAL)
	@echo "  LINK  " $@
	@$(LINK) $(LDFLAGS) -o $@ -Wl,--whole-archive $^ -Wl,--no-whole-archive $(LINK_EXTERNAL) $(BOOST_UNIT_TEST_FRAMEWORK) 

.SECONDEXPANSION:

STEM := %

%.a:	$$(patsubst $$(STEM).cpp,build/$$(STEM).o,$$(SRCS_$$*))
	@echo "  LIB   " `basename \`pwd\``/$@
	@rm -f $@
	@$(AR) rcs $@ $^

%.so:	$$(patsubst $$(STEM).cpp,build/$$(STEM).o,$$(SRCS_$$*)) $(LINK_INTERNAL)
	@echo "  LINK  " `basename \`pwd\``/$@
	@$(LINK) $(LDFLAGS) -shared -o $@ $^ -rdynamic $(LINK_EXTERNAL)

$(PROGS):	build/$$@.o $$(patsubst %.cpp,build/%.o,$$(SRCS_$$@)) $(LIBS) $(LINK_INTERNAL)
	@echo "  LINK  " $@
	@$(LINK) $(LDFLAGS) -o $@ $^ -rdynamic $(LINK_EXTERNAL)
