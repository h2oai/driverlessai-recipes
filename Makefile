.PHONY: all

all:
	mkdir -p base_templates
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/transformer_utils.py | sed '1d; $$d' > base_templates/custom_transformer.py
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/models.py | sed '1d; $$d' > base_templates/custom_model.py
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/metrics.py | sed '1d; $$d' > base_templates/custom_scorer.py
	python ./gen-readme.py > README.md
