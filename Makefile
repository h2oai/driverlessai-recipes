.PHONY: README.md FAQ.md

all: README.md FAQ.md

README.md:
	python ./gen-readme.py > README.md

FAQ.md: get_code_fragments
	python ./gen-faq.py > FAQ.md
	rm FAQ.*.py

get_code_fragments:
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/transformer_utils.py | sed '1d; $$d' > FAQ.transformers.py
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/models.py | sed '1d; $$d' > FAQ.models.py
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/metrics.py | sed '1d; $$d' > FAQ.scorers.py
