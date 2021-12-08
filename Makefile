.PHONY: all

all:
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/transformer_utils.py | sed '1d; $$d' > transformers/transformer_template.py
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/models_custom.py | sed '1d; $$d' > models/model_template.py
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/metrics.py | sed '1d; $$d' > scorers/scorer_template.py
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/data.py | sed '1d; $$d' > data/data_template.py
	mkdir -p individuals/
	sed -n '/# START_CUSTOM/,/# END_CUSTOM/p' ../h2oai/h2oaicore/ga.py | sed '1d; $$d' > individuals/individual_template.py
	python ./gen-readme.py > README.md
	python ./livecode/gen-readme.py > ./livecode/README.md
