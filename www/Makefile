.PHONY: www clean dev up

# banner
Q ?= echo

PYTHON ?= python3

ALL_CUSTOM_CODE := $(shell find ../models ../scorers ../transformers -type f -name "*.py")

build: src/content/recipes.yaml
	@gatsby build

clean:
	@gatsby clean

dev:
	@gatsby develop

up: clean build
	@$(Q) aws s3 sync public/ s3://catalog.h2o.ai/ --acl public-read

src/content/recipes.yaml: $(ALL_CUSTOM_CODE)
	(cd ../ && $(PYTHON) gen-meta.py) > $@
