.DEFAULT_GOAL := install
SHELL := /bin/bash

PY_VERSION ?= 3.5
VENV_PATH ?= "venv"
REQUIREMENTS ?= requirements.txt
ACTIVATE_VENV = . $(VENV_PATH)/bin/activate
PIP ?= pip$(PY_VERSION)
PYVENV ?= pyvenv-$(PY_VERSION)
IMG_DIR ?= img


install: | make-dirs install-py-deps

make-dirs:
	$(PYVENV) $(VENV_PATH)
	ln -s ../results data
	mkdir $(IMG_DIR)

install-py-deps:
	$(ACTIVATE_VENV) ; \
	$(PIP) install --upgrade pip ; \
	$(PIP) install -r $(REQUIREMENTS) ; \
	deactivate ; 

clean:
	@rm -f *.png *.jpg
	@rm -rf venv
	@rm -f data
	@rm -rf $(IMG_DIR)
