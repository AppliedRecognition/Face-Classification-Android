INCLUDES	=
PROGS_INTERNAL	= tools raw_image_io
PROGS_EXTERNAL	:= $(shell grep "^PROGS_EXTERNAL" */Makefile |sed 's/.*=//')
PROJECT_BASE	= ..

all:	$(LINK_INTERNAL) test_all

include Makefile.inc

test_all: applog/unit_tests.o $(LINK_INTERNAL)
	@for i in $(LINK_INTERNAL); do $(MAKE) -s -C `dirname $$i` tests; done
	@echo "  LINK  " $@
	@$(LINK) $(LDFLAGS) -o $@ $(wildcard */build/tests/*.o) $^ $(LINK_EXTERNAL) $(BOOST_UNIT_TEST_FRAMEWORK)

allclean:
	@for i in $(LINK_INTERNAL); do $(MAKE) -s -C `dirname $$i` clean; done
	@echo "  CLEAN " `basename \`pwd\``
	@rm -f test_all
