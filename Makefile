.PHONY: all

all:
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/transformer_utils.py | sed '1d; $$d' > transformers/transformer_template.py
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/models.py | sed '1d; $$d' > models/model_template.py
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/metrics.py | sed '1d; $$d' > scorers/scorer_template.py
	python ./gen-readme.py > README.md
