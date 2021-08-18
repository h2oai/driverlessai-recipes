from h2oaicore.models import TextBERTModel, CustomModel


class PortugueseBertModel(TextBERTModel, CustomModel):
    """
    Custom model class for using a model from https://huggingface.co/models.
    Ensure that the model can be loaded using AutoModelForSequenceClassification.from_pretrained(_model_name)

    The class inherits :
      - CustomModel that really is just a tag. It's there to make sure DAI knows it's a custom model.
      - TextBERTModel so that the custom model inherits all the properties and methods.
    How to use:
        - Disable genetic algorithm in the expert settings.
        - Enable PortugueseBertModel in the expert settings and disable all other models.
        - One model is fitted on the whole training dataset. To be able to see model scores, specify a validation or
          a test dataset.
    """
    _mojo = False
    _booster_str = "pytorch_custom"
    _model_name = "bert-base-portuguese-cased"

    def _set_model_name(self, *args, **kwargs):
        self.model_path = self.model_name = self._model_name
