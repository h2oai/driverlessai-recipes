"""
Custom Bert model pretrained on Portuguese.
"""
import os
import shutil
from urllib.parse import urlparse

import requests

from h2oaicore.models import TextBERTModel, CustomModel
from h2oaicore.systemutils import make_experiment_logger, temporary_files_path, atomic_move, loggerinfo


def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


def maybe_download_language_model(logger,
                                  save_directory,
                                  model_link,
                                  config_link,
                                  vocab_link):
    model_name = "pytorch_model.bin"
    if isinstance(model_link, str):
        model_name = model_link.split('/')[-1]
        if '.bin' not in model_name:
            model_name = "pytorch_model.bin"

    maybe_download(url=config_link,
                   dest=os.path.join(save_directory, "config.json"),
                   logger=logger)
    maybe_download(url=vocab_link,
                   dest=os.path.join(save_directory, "vocab.txt"),
                   logger=logger)
    maybe_download(url=model_link,
                   dest=os.path.join(save_directory, model_name),
                   logger=logger)


def maybe_download(url, dest, logger=None):
    if not is_url(url):
        loggerinfo(logger, f"{url} is not a valid URL.")
        return

    dest_tmp = dest + ".tmp"
    if os.path.exists(dest):
        loggerinfo(logger, f"already downloaded {url} -> {dest}")
        return

    if os.path.exists(dest_tmp):
        loggerinfo(logger, f"Download has already started {url} -> {dest_tmp}. "
        f"Delete {dest_tmp} to download the file once more.")
        return

    loggerinfo(logger, f"Downloading {url} -> {dest}")
    url_data = requests.get(url, stream=True)
    if url_data.status_code != requests.codes.ok:
        msg = "Cannot get url %s, code: %s, reason: %s" % (
            str(url), str(url_data.status_code), str(url_data.reason))
        raise requests.exceptions.RequestException(msg)
    url_data.raw.decode_content = True
    if not os.path.isdir(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest_tmp, 'wb') as f:
        shutil.copyfileobj(url_data.raw, f)

    atomic_move(dest_tmp, dest)


def check_correct_name(custom_name):
    allowed_pretrained_models = ['bert', 'openai-gpt', 'gpt2', 'transfo-xl', 'xlnet', 'xlm-roberta',
                                 'xlm', 'roberta', 'distilbert', 'camembert', 'ctrl', 'albert']
    assert len([model_name for model_name in allowed_pretrained_models
                if model_name in custom_name]), f"{custom_name} needs to contain the name" \
        " of the pretrained model architecture (e.g. bert or xlnet) " \
        "to be able to process the model correctly."


class CustomBertModel(TextBERTModel, CustomModel):
    """
    Custom model class for using pretrained transformer models.
    The class inherits :
      - CustomModel that really is just a tag. It's there to make sure DAI knows it's a custom model.
      - TextBERTModel so that the custom model inherits all the properties and methods.

    Supported model architecture:
    'bert', 'openai-gpt', 'gpt2', 'transfo-xl', 'xlnet', 'xlm-roberta',
    'xlm', 'roberta', 'distilbert', 'camembert', 'ctrl', 'albert'

    How to use:
        - You have already downloaded the weights, the vocab and the config file:
            - Set _model_path as the folder where the weights, the vocab and the config file are stored.
            - Set _model_name according to the pretrained architecture (e.g. bert-base-uncased).
        - You want to to download the weights, the vocab and the config file:
            - Set _model_link, _config_link and _vocab_link accordingly.
            - _model_path is the folder where the weights, the vocab and the config file will be saved.
            - Set _model_name according to the pretrained architecture (e.g. bert-base-uncased).

        - Important:
          _model_path needs to contain the name of the pretrained model architecture (e.g. bert or xlnet)
          to be able to load the model correctly.

        - Disable genetic algorithm in the expert setting.
    """

    # _model_path is the full path to the directory where the weights, vocab and the config will be saved.
    _model_name = NotImplemented  # Will be used to create the MOJO
    _model_path = NotImplemented

    _model_link = NotImplemented
    _config_link = NotImplemented
    _vocab_link = NotImplemented
    _booster_str = "pytorch-custom"

    # Requirements for MOJO creation:
    # _model_name needs to be  one of
    # bert-base-uncased, bert-base-multilingual-cased, xlnet-base-cased, roberta-base, distilbert-base-uncased
    # vocab.txt needs to be the same as vocab.txt used in _model_name (no custom vocabulary yet).
    _mojo = False

    @staticmethod
    def is_enabled():
        return False  # Abstract Base model should not show up in models.

    def _set_model_name(self, language_detected):
        self.model_path = self.__class__._model_path
        self.model_name = self.__class__._model_name
        check_correct_name(self.model_path)
        check_correct_name(self.model_name)

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id, tmp_dir=self.context.tmp_dir,
                                            experiment_tmp_dir=self.context.experiment_tmp_dir)
        maybe_download_language_model(logger,
                                      save_directory=self.__class__._model_path,
                                      model_link=self.__class__._model_link,
                                      config_link=self.__class__._config_link,
                                      vocab_link=self.__class__._vocab_link)
        super().fit(X, y, sample_weight, eval_set, sample_weight_eval_set, **kwargs)


class PortugueseBertModel(CustomBertModel):
    _model_name = "bert-base-portuguese-cased"
    _model_path = os.path.join(temporary_files_path, "portuguese_bert_language_model/")

    # Hugging Face model for Portuguese language found here: https://huggingface.co/neuralmind/bert-base-portuguese-cased/tree/main
    _model_link = "https://huggingface.co/neuralmind/bert-base-portuguese-cased/blob/main/pytorch_model.bin"
    _config_link = "https://huggingface.co/neuralmind/bert-base-portuguese-cased/blob/main/config.json"
    _vocab_link = "https://huggingface.co/neuralmind/bert-base-portuguese-cased/blob/main/vocab.txt"

    _mojo = True

    @staticmethod
    def is_enabled():
        return True


