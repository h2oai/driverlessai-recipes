"""Convert a path to an image (JPG/JPEG/PNG) to a vector of class probabilities created by a pretrained ImageNet deeplearning model (Keras, TensorFlow)."""
import importlib
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.models import TensorFlowModel
import datatable as dt
import numpy as np
from h2oaicore.systemutils import small_job_pool, user_dir, dummypool, print_debug, remove
import requests
import shutil
import uuid
import os


class MyImgTransformer(TensorFlowModel, CustomTransformer):
    # Need Pillow before nlp imports keras, else when here too late.
    # I.e. wasn't enough to put keras imports inside fit/transform to delay after Pillow installed
    _modules_needed_by_name = ['pillow==8.3.2']
    _tensorflow = True
    _mojo = False
    _parallel_task = True  # assumes will use n_jobs in params_base
    _can_use_gpu = True
    _can_use_multi_gpu = True
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def do_acceptance_test():
        return True  # False

    @staticmethod
    def acceptance_test_coverage_fraction():
        import os
        return 0.05 if 'GIT_HASH' in os.environ else 1.0

    @staticmethod
    def get_default_properties():
        return dict(col_type="image", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def enabled_setting():
        return 'on'

    def __init__(self, batch_size=32, **kwargs):
        CustomTransformer.__init__(self, **kwargs)
        TensorFlowModel.__init__(self, **kwargs)
        self.uuid_tmp = str(uuid.uuid4())[:6]
        self.experiment_id = self.__class__.__name__ + self.uuid_tmp
        # super().__init__(**kwargs)
        self.batch_size = batch_size
        self.model_name = "resnet_keras.h5p"
        self.uuid = "%s-img-data-" % self.__class__.__name__ + self.model_name  # + str(uuid.uuid4())[:6] # no, keeps changing and re-loadeing every init
        self.model_path = os.path.join(user_dir(), self.uuid + ".model")
        self.model_tmp_path = self.model_path + "_" + self.uuid_tmp + ".tmp"
        if not os.path.exists(self.model_path):
            self.download(
                url="http://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/recipes/transformers/img/%s" % self.model_name,
                dest=self.model_path)
        with open(self.model_path, 'rb') as f:
            self.model_bytes = f.read()
            # remove(self.model_path) # avoid re-downloads

    def atomic_move(self, src, dst):
        try:
            shutil.move(src, dst)
        except FileExistsError:
            pass
        remove(src)

    def download(self, url, dest):
        if os.path.exists(dest):
            print("already downloaded %s -> %s" % (url, dest))
            return
        print("downloading %s to %s" % (url, dest))
        url_data = requests.get(url, stream=True)
        if url_data.status_code != requests.codes.ok:
            msg = "Cannot get url %s, code: %s, reason: %s" % (
                str(url), str(url_data.status_code), str(url_data.reason))
            raise requests.exceptions.RequestException(msg)
        url_data.raw.decode_content = True
        if not os.path.isdir(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        uuid_tmp = str(uuid.uuid4())[:6]
        dest_tmp = dest + "_" + uuid_tmp + ".tmp"
        with open(dest_tmp, 'wb') as f:
            shutil.copyfileobj(url_data.raw, f)
        self.atomic_move(dest_tmp, dest)

    @property
    def display_name(self):
        return "MyImgTransformerBatchSize%d" % self.batch_size

    @staticmethod
    def get_parameter_choices():
        return dict(batch_size=[16, 32, 64])

    @staticmethod
    def get_default_properties():
        return dict(col_type="categorical", min_cols=1, max_cols=1, relative_importance=1)

    def preprocess_image(self, source_img_path, check_only=False):
        try:
            final_img_path = os.path.join(user_dir(), self.uuid, os.path.basename(source_img_path))
        except:  # we are sometimes getting np.float32, why?
            return None
        delete = False
        if not os.path.exists(final_img_path):
            if not os.path.exists(source_img_path):
                try:
                    self.download(source_img_path, final_img_path)
                except requests.RequestException as e:
                    # print_debug("Error: %s for source_img_path: %s" % (str(e), str(source_img_path)))
                    return None
                delete = False  # True to avoid re-download or a race condition between multiple procs
            else:
                final_img_path = source_img_path
        if not check_only:
            import h2oaicore.keras as keras
            importlib.reload(keras)
            img = keras.preprocessing.image.load_img(final_img_path, target_size=(224, 224))
            if delete:
                remove(final_img_path)
            x = keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = keras.applications.resnet50.preprocess_input(x)
            return x
        else:
            return True

    def fit_transform(self, X: dt.Frame, y: np.array = None, **kwargs):
        return self.transform(X, **kwargs)

    def transform(self, X: dt.Frame, **kwargs):
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                f.write(self.model_bytes)

        # remove(self.model_path) # can't remove, used by other procs or later
        self.col_name = self.input_feature_names[0]
        values = X[:, self.col_name].to_numpy().ravel()
        self.batch_size = min(len(values), self.batch_size)
        values_ = np.array_split(values, int(len(values) / self.batch_size) + 1)
        print(values_)

        # check if data is image related
        results = []
        for v in values_:
            images = []
            for x in v:
                if True or x[-4:] in [".jpg", ".png", ".jpeg"]:
                    image = self.preprocess_image(x, check_only=True)
                    images.append(image)
                else:
                    raise NotImplementedError
            # deal with missing images (None in images)
            images = [x for x in images if x is not None]
            results.extend(images)

        if len(results) > 0:
            # don't use GPU memory unless actually found relevant data
            import h2oaicore.keras as keras
            # self.tf_config = self.set_tf_config(kwargs)
            self.tf_config = self.ConfigProto()
            # self.tf_config.gpu_options.allow_growth = True
            self.tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
            keras.backend.set_session(session=TensorFlowModel.make_sess(self.tf_config))
            # importlib.reload(keras)
            self.model = keras.models.load_model(self.model_path)

            results = []
            for v in values_:
                images = []
                for x in v:
                    if True or x[-4:] in [".jpg", ".png", ".jpeg"]:
                        image = self.preprocess_image(x)
                        images.append(image)
                    else:
                        raise NotImplementedError
                # deal with missing images (None in images)
                good_imagei = None
                for imagei, image in enumerate(images):
                    if image is not None:
                        good_imagei = imagei
                        break
                if len(images) > 0:
                    msg = "no good images out of %d images" % len(images)
                    if False:
                        assert good_imagei is not None, msg
                    elif good_imagei is None:
                        pass
                        # print_debug(msg)
                if good_imagei is not None:
                    for imagei, image in enumerate(images):
                        if image is None:
                            images[imagei] = images[good_imagei] * 0  # impute 0 for missing images
                    images = np.vstack(images)
                    results.append(self.model.predict(images))
        if len(results) > 0:
            return dt.Frame(np.vstack(results))
        else:
            return dt.Frame([0] * X.shape[0])
