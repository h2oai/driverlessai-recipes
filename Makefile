.PHONY: README.md FAQ.md

all: README.md FAQ.md

README.md:
	python ./gen-readme.py > README.md

FAQ.md: get_code_fragments
	python ./gen-faq.py > FAQ.md
	#rm FAQ.*.py

get_code_fragments:
	mkdir -p base_classes
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/transformer_utils.py | sed '1d; $$d' > base_classes/custom_transformer.py
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/models.py | sed '1d; $$d' > base_classes/custom_model.py
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/metrics.py | sed '1d; $$d' > base_classes/custom_scorer.py
