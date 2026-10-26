# Thin wrapper around tools/hp.py; on Windows run the python commands directly.
PY ?= python

.PHONY: setup test build nhats dist all guard
setup: ; $(PY) tools/hp.py setup
test: ; $(PY) tools/hp.py test
build: ; $(PY) tools/hp.py build
dist: ; $(PY) tools/hp.py dist
all: ; $(PY) tools/hp.py all
guard: ; $(PY) tools/guard_data.py
nhats: ; $(PY) tools/hp.py nhats --data-dir "$(NHATS_DIR)"
