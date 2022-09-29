""" Calibrated Classifier Model: To calibrate predictions using Platt's scaling, Isotonic Regression or Splines
"""

import copy
import datatable as dt
from h2oaicore.mojo import MojoWriter, MojoFrame
from h2oaicore.systemutils import config
from h2oaicore.models import CustomModel, LightGBMModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.special import softmax, expit
from sklearn.calibration import CalibratedClassifierCV


class SklearnWrapper:  # to trick CalibratedClassifierCV from sklearn
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_simple_base(X)

    def fit(X, y):  # SKLearn checks if this method exists in Estimator
        pass


class CalibratedClassifierModel:
    _regression = False
    _binary = True
    _multiclass = True
    _can_use_gpu = True
    _mojo = True
    _description = "Calibrated Classifier Model (LightGBM)"
    _supports_predict_shuffle_scoring = False

    le = LabelEncoder()

    _modules_needed_by_name = ['ml_insights==0.1.4']  # for SplineCalibration

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def do_acceptance_test():
        """
        Return whether to do acceptance tests during upload of recipe and during start of Driverless AI.
        Acceptance tests perform a number of sanity checks on small data, and attempt to provide helpful instructions
        for how to fix any potential issues. Disable if your recipe requires specific data or won't work on random data.
        """
        return True

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        assert len(self.__class__.__bases__) == 3
        assert CalibratedClassifierModel in self.__class__.__bases__

        self.le.fit(self.labels)
        y_ = self.le.transform(y)

        whoami = [x for x in self.__class__.__bases__ if (x != CustomModel and x != CalibratedClassifierModel)][0]

        kwargs_classification = copy.deepcopy(self.params_base)
        kwargs_update = dict(
            num_classes=len(self.le.classes_),
            labels=list(np.arange(len(self.le.classes_))),
        )
        kwargs_classification.update(kwargs_update)
        for k in kwargs:
            if k in kwargs_classification:
                kwargs[k] = kwargs_classification[k]

        model_classification = whoami(context=self.context,
                                      unfitted_pipeline_path=self.unfitted_pipeline_path,
                                      transformed_features=self.transformed_features,
                                      original_user_cols=self.original_user_cols,
                                      date_format_strings=self.date_format_strings, **kwargs_classification)

        eval_set_classification = None
        val_y = None
        calib_perc = self.params.get("calib_perc", .1)
        if eval_set is not None:
            eval_set_y = self.le.transform(eval_set[0][1])
            eval_set_y_raw = eval_set[0][1]
            val_y = eval_set_y.astype(int)
            eval_set_classification = [(eval_set[0][0], val_y)]

        if not self.params["use_validation"] or eval_set is None:
            # Stratified split with classes control - making sure all classes present in both train and test
            unique_cls = np.unique(y_)
            tr_indx, te_indx = [], []

            for c in unique_cls:
                c_indx = np.argwhere(y_ == c).ravel()
                indx = np.random.permutation(c_indx)
                if self.params["calib_method"] in ["sigmoid", "isotonic"]:
                    start_indx = max(1, int(calib_perc * len(c_indx)))
                else:
                    start_indx = max(3, int(calib_perc * len(c_indx)))

                tr_indx += list(indx[start_indx:])
                te_indx += list(indx[:start_indx])
            tr_indx = np.array(tr_indx)
            te_indx = np.array(te_indx)

            X_train, y_train = X[tr_indx, :], y_.astype(int)[tr_indx]
            if self.params["calib_method"] in ["sigmoid", "isotonic"]:
                X_calibrate, y_calibrate = X[te_indx, :].to_pandas(), y[te_indx].ravel()
            else:
                X_calibrate, y_calibrate = X[te_indx, :].to_pandas(), y_.astype(int)[te_indx].ravel()

            if sample_weight is not None:
                sample_weight_ = sample_weight[tr_indx]
                sample_weight_calib = sample_weight[te_indx]
            else:
                sample_weight_ = sample_weight
                sample_weight_calib = sample_weight
        else:
            X_train, y_train = X, y_.astype(int)
            X_calibrate, y_calibrate = eval_set_classification[0]
            if self.params["calib_method"] in ["sigmoid", "isotonic"]:
                y_calibrate = eval_set_y_raw
            sample_weight_ = sample_weight
            sample_weight_calib = None if sample_weight_eval_set is None else sample_weight_eval_set[0]

        # mimic rest of fit_base not done:
        # get self.observed_labels
        model_classification.check_labels_and_response(y_train, val_y=val_y)
        model_classification.orig_cols = self.orig_cols
        model_classification.X_shape = self.X_shape

        model_classification.fit(X_train, y_train,
                                 sample_weight=sample_weight_, eval_set=eval_set_classification,
                                 sample_weight_eval_set=sample_weight_eval_set, **kwargs)

        model_classification.fitted = True
        model_classification.eval_set_used_during_fit = val_y is not None

        # calibration
        sk_model = SklearnWrapper(model_classification)
        sk_model.classes_ = self.le.classes_
        sk_model.fitted = True
        sk_model.eval_set_used_during_fit = val_y is not None

        # model_classification.predict_proba = model_classification.predict_simple_base
        # model_classification.classes_ = self.le.classes_
        if self.params["calib_method"] in ["sigmoid", "isotonic"]:
            calibrator = CalibratedClassifierCV(
                base_estimator=sk_model,
                method=self.params["calib_method"],
                cv='prefit', ensemble=False)

            calibrator.fit(X_calibrate, y_calibrate, sample_weight=sample_weight_calib)

            self.calib_method = calibrator.method

            if calibrator.method == "sigmoid":
                self.slope = []
                self.intercept = []

                for c in calibrator.calibrated_classifiers_[0].calibrators_:
                    self.slope.append(c.a_)
                    self.intercept.append(c.b_)

            elif calibrator.method == "isotonic":
                self._necessary_X_ = []
                self._necessary_y_ = []

                self.X_min_ = []
                self.X_max_ = []
                for c in calibrator.calibrated_classifiers_[0].calibrators_:
                    self._necessary_X_.append(c.X_thresholds_)
                    self._necessary_y_.append(c.y_thresholds_)

                    self.X_min_.append(c.X_min_)
                    self.X_max_.append(c.X_max_)


            else:
                raise RuntimeError('Unknown calibration method in fit()')

        elif self.params["calib_method"] in ["spline"]:
            import ml_insights as mli
            self.calib_method = "spline"
            spline = mli.SplineCalib(
                penalty='l2',
                solver='liblinear',
                reg_param_vec='default',
                cv_spline=3, random_state=4451,
                knot_sample_size=30,
            )

            preds = sk_model.predict_proba(X_calibrate)

            for c in range(preds.shape[1]):
                if len(np.unique(preds[:, c])) < 3:  # we need at least 3 unique points to form the knots
                    preds[:, c] = preds[:, c] + .0001 * np.random.randn(len(preds[:, c]))

            spline.fit(preds, y_calibrate, verbose=False)  # no weight support so far :(

            self.calib_logodds_scale = spline.logodds_scale
            self.calib_logodds_eps = spline.logodds_eps

            self.calib_knot_vec_tr = []
            self.calib_basis_coef_vec = []

            if spline.n_classes > 2:
                for calib_ in spline.binary_splinecalibs:
                    self.calib_knot_vec_tr.append(calib_.knot_vec_tr)
                    self.calib_basis_coef_vec.append(calib_.basis_coef_vec)
            else:
                self.calib_knot_vec_tr.append(spline.knot_vec_tr)
                self.calib_basis_coef_vec.append(spline.basis_coef_vec)

        else:
            raise RuntimeError('Unknown calibration method in fit()')
        # calibration

        varimp = model_classification.imp_features(columns=X.names)[['LGain', 'LInteraction']].dropna(axis=0)
        varimp.index = varimp['LInteraction']
        varimp = varimp['LGain']
        varimp = varimp[:len(X.names)]
        varimp = varimp.reindex(X.names).values
        importances = varimp

        iters = model_classification.best_iterations
        iters = int(max(1, iters))
        self.set_model_properties(model=model_classification.model,
                                  features=list(X.names), importances=importances, iterations=iters
                                  )

    @staticmethod
    def _natural_cubic_spline_basis_expansion(xpts, knots):
        num_knots = len(knots)
        num_pts = len(xpts)
        outmat = np.zeros((num_pts, num_knots))
        outmat[:, 0] = np.ones(num_pts)
        outmat[:, 1] = xpts

        # last knot calc
        denom = knots[-1] - knots[-2]
        numer = (np.maximum(xpts - knots[-2], np.zeros(num_pts)) ** 3 -
                 np.maximum(xpts - knots[-1], np.zeros(num_pts)) ** 3)
        last_knot = numer / denom

        # current knots calc
        for i in range(1, num_knots - 1):
            denom = knots[-1] - knots[i - 1]
            numer = (np.maximum(xpts - knots[i - 1], np.zeros(num_pts)) ** 3 -
                     np.maximum(xpts - knots[-1], np.zeros(num_pts)) ** 3)
            outmat[:, i + 1] = (numer / denom) - last_knot

        return outmat

    def predict(self, X, **kwargs):
        from scipy import interpolate

        X = dt.Frame(X)
        model, _, _, _ = self.get_model_properties()
        preds = model.predict_proba(X)

        if preds.shape[1] <= 2:
            if self.calib_method == "sigmoid":
                scaled_preds = expit(-(self.slope[0] * preds[:, 1] + self.intercept[0]))

            elif self.calib_method == "isotonic":
                f_ = interpolate.interp1d(
                    self._necessary_X_[0],
                    self._necessary_y_[0],
                    kind='linear',
                    bounds_error='nan'
                )
                scaled_preds = f_(np.clip(preds[:, 1], self.X_min_[0], self.X_max_[0]))

            elif self.calib_method == "spline":
                y_in_to_use = preds[:, 1]
                if self.calib_logodds_scale:
                    y_in_to_use = np.minimum(1 - self.calib_logodds_eps, y_in_to_use)
                    y_in_to_use = np.maximum(self.calib_logodds_eps, y_in_to_use)
                    y_model_tr = np.log(y_in_to_use / (1 - y_in_to_use))
                else:
                    y_model_tr = y_in_to_use
                scaled_preds = self._natural_cubic_spline_basis_expansion(y_model_tr, self.calib_knot_vec_tr[0])
                scaled_preds = scaled_preds.dot(self.calib_basis_coef_vec[0].T)
                scaled_preds = 1 / (1 + np.exp(-scaled_preds))
                scaled_preds = scaled_preds.ravel()
            else:
                raise RuntimeError('Unknown calibration method in predict()')
            preds[:, 1] = scaled_preds
            preds[:, 0] = 1. - scaled_preds
        else:
            for c in range(preds.shape[1]):
                if self.calib_method == "sigmoid":
                    scaled_preds = expit(-(self.slope[c] * preds[:, c] + self.intercept[c]))
                elif self.calib_method == "isotonic":
                    f_ = interpolate.interp1d(
                        self._necessary_X_[c],
                        self._necessary_y_[c],
                        kind='linear',
                        bounds_error='nan'
                    )
                    scaled_preds = f_(np.clip(preds[:, c], self.X_min_[c], self.X_max_[c]))

                elif self.calib_method == "spline":
                    y_in_to_use = preds[:, c]
                    if self.calib_logodds_scale:
                        y_in_to_use = np.minimum(1 - self.calib_logodds_eps, y_in_to_use)
                        y_in_to_use = np.maximum(self.calib_logodds_eps, y_in_to_use)
                        y_model_tr = np.log(y_in_to_use / (1 - y_in_to_use))
                    else:
                        y_model_tr = y_in_to_use
                    scaled_preds = self._natural_cubic_spline_basis_expansion(y_model_tr, self.calib_knot_vec_tr[c])
                    scaled_preds = scaled_preds.dot(self.calib_basis_coef_vec[c].T)
                    scaled_preds = 1 / (1 + np.exp(-scaled_preds))
                    scaled_preds = scaled_preds.ravel()
                else:
                    raise RuntimeError('Unknown calibration method in predict()')

                preds[:, c] = scaled_preds
            preds = preds / np.sum(preds, 1).reshape(-1, 1)
        return preds


from h2oaicore.mojo import MojoWriter, MojoFrame


class CalibratedClassifierLGBMModel(CalibratedClassifierModel, LightGBMModel, CustomModel):
    _mojo = True

    @property
    def has_pred_contribs(self):
        return False

    @property
    def has_output_margin(self):
        return False

    def set_default_params(self, **kwargs):
        super().set_default_params(**kwargs)
        # To activate
        # config_overrides = "recipe_dict=\"{'calibrationModel_use_validation': True}\""
        self.params["use_validation"] = config.recipe_dict.get('calibrationModel_use_validation', False)
        if not self.params["use_validation"]:
            self.params["calib_perc"] = .1
        self.params["calib_method"] = "sigmoid"

    def mutate_params(self, **kwargs):
        super().mutate_params(**kwargs)
        # To activate
        # config_overrides = "recipe_dict=\"{'calibrationModel_use_validation': True}\""
        self.params["use_validation"] = config.recipe_dict.get('calibrationModel_use_validation', False)
        if not self.params["use_validation"]:
            self.params["calib_perc"] = np.random.choice([.05, .1, .15, .2])

        methods = ["isotonic", "sigmoid"]

        import importlib
        mli_spec = importlib.util.find_spec("ml_insights")
        found = mli_spec is not None
        if found:
            methods += ["spline"]

        self.params["calib_method"] = np.random.choice(methods)

    def write_to_mojo(self, mojo: MojoWriter, iframe: MojoFrame, group_uuid=None, group_name=None):
        return self.to_mojo(mojo=mojo, iframe=iframe, group_uuid=group_uuid, group_name=group_name)

    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame, group_uuid=None, group_name=None):

        from h2oaicore.mojo import MojoColumn
        from h2oaicore.mojo_transformers import (MjT_ConstBinaryOp, MjT_Sigmoid, MjT_AsType,
                                                 MjT_Agg, MjT_BinaryOp, MjT_IntervalMap, MjT_Clip, MjT_Log)
        import uuid
        group_uuid = str(uuid.uuid4())
        group_name = self.__class__.__name__

        _iframe = super().write_to_mojo(mojo=mojo, iframe=iframe, group_uuid=group_uuid, group_name=group_name)
        res = MojoFrame()

        def _get_new_pair(left, right):
            pair = MojoFrame()
            pair.cbind(left)
            pair.cbind(right)
            return pair

        for c in range(len(_iframe)):
            icol = _iframe.get_column(c)

            def _get_new_col(name, type_=None):
                ocol_ = MojoColumn(name=name, dtype=icol.type if type_ is None else type_)
                oframe_ = MojoFrame(columns=[ocol_])
                return oframe_

            if self.calib_method == "sigmoid":

                oframe1 = _get_new_col(icol.name + "_slope")
                oframe2 = _get_new_col(icol.name + "_intercept")
                oframe3 = _get_new_col(icol.name + "_negative")
                oframe4 = _get_new_col(icol.name + "_calibrated", type_="float64")
                oframe5 = _get_new_col(icol.name + "_astype")

                mojo += MjT_ConstBinaryOp(iframe=_iframe[c], oframe=oframe1, op="multiply", const=self.slope[c],
                                          pos="right",
                                          group_uuid=group_uuid, group_name=group_name)

                mojo += MjT_ConstBinaryOp(iframe=oframe1, oframe=oframe2, op="add", const=self.intercept[c],
                                          pos="right",
                                          group_uuid=group_uuid, group_name=group_name)

                mojo += MjT_ConstBinaryOp(iframe=oframe2, oframe=oframe3, op="multiply", const=-1., pos="right",
                                          group_uuid=group_uuid, group_name=group_name)

                mojo += MjT_Sigmoid(iframe=oframe3, oframe=oframe4, group_uuid=group_uuid, group_name=group_name)
                mojo += MjT_AsType(iframe=oframe4, oframe=oframe5, type="float32", group_uuid=group_uuid,
                                   group_name=group_name)

                res.cbind(oframe5)
            elif self.calib_method == "isotonic":
                X = list(self._necessary_X_[c])
                y = list(self._necessary_y_[c])
                if len(y) == 1:
                    oframe1 = _get_new_col(icol.name + "_zeroing")
                    new_y = _get_new_col(icol.name + "_addingConst")

                    mojo += MjT_ConstBinaryOp(iframe=_iframe[c], oframe=oframe1, op="multiply", const=0, pos="right",
                                              group_uuid=group_uuid, group_name=group_name)

                    mojo += MjT_ConstBinaryOp(iframe=oframe1, oframe=new_y, op="add", const=y[0], pos="right",
                                              group_uuid=group_uuid, group_name=group_name)

                else:
                    max_X = X + [self._necessary_X_[c][-1], None]
                    min_X = [self._necessary_X_[c][0]] + X + [None]

                    max_y = y + [self._necessary_y_[c][-1], None]
                    min_y = [self._necessary_y_[c][0]] + y + [None]

                    ocol1 = MojoColumn(name=icol.name + "_maxX", dtype=icol.type)
                    ocol2 = MojoColumn(name=icol.name + "_minX", dtype=icol.type)
                    ocol3 = MojoColumn(name=icol.name + "_maxY", dtype=icol.type)
                    ocol4 = MojoColumn(name=icol.name + "_minY", dtype=icol.type)
                    XY = MojoFrame(columns=[ocol1, ocol2, ocol3, ocol4])

                    # clipping
                    inp_clipped = _get_new_col(icol.name + "_clipped")
                    mojo += MjT_Clip(iframe=_iframe[c], oframe=inp_clipped,
                                     min=self.X_min_[c], max=self.X_max_[c],
                                     group_uuid=group_uuid, group_name=group_name
                                     )

                    # search for coordinates
                    mojo += MjT_IntervalMap(
                        iframe=inp_clipped, oframe=XY,
                        breakpoints=X,
                        values=[[x1, x0, y1, y0] for x1, x0, y1, y0 in zip(max_X, min_X, max_y, min_y)],
                        group_uuid=group_uuid, group_name=group_name
                    )

                    # interpolation
                    curr_diff = _get_new_col(icol.name + "_currDiff")
                    pair = _get_new_pair(inp_clipped, XY[1])
                    mojo += MjT_BinaryOp(iframe=pair, oframe=curr_diff, op="subtract", group_uuid=group_uuid,
                                         group_name=group_name)

                    y_diff = _get_new_col(icol.name + "_yDiff")
                    pair = _get_new_pair(XY[2], XY[3])
                    mojo += MjT_BinaryOp(iframe=pair, oframe=y_diff, op="subtract", group_uuid=group_uuid,
                                         group_name=group_name)

                    X_diff = _get_new_col(icol.name + "_XDiff")
                    pair = _get_new_pair(XY[0], XY[1])
                    mojo += MjT_BinaryOp(iframe=pair, oframe=X_diff, op="subtract", group_uuid=group_uuid,
                                         group_name=group_name)

                    xy_ratio = _get_new_col(icol.name + "_xyRatio")
                    pair = _get_new_pair(y_diff, X_diff)
                    mojo += MjT_BinaryOp(iframe=pair, oframe=xy_ratio, op="divide", eps=1e-10, group_uuid=group_uuid,
                                         group_name=group_name)

                    scaled_cur_diff = _get_new_col(icol.name + "_scaledCurDiff")
                    pair = _get_new_pair(xy_ratio, curr_diff)
                    mojo += MjT_BinaryOp(iframe=pair, oframe=scaled_cur_diff, op="multiply", group_uuid=group_uuid,
                                         group_name=group_name)

                    new_y = _get_new_col(icol.name + "_newY")
                    pair = _get_new_pair(XY[3], scaled_cur_diff)
                    mojo += MjT_BinaryOp(iframe=pair, oframe=new_y, op="add", group_uuid=group_uuid,
                                         group_name=group_name)

                res.cbind(new_y)

            elif self.calib_method == "spline":
                if self.calib_logodds_scale:
                    oframe1 = _get_new_col(icol.name + "_clipped")
                    mojo += MjT_Clip(iframe=_iframe[c], oframe=oframe1,
                                     min=self.calib_logodds_eps, max=1 - self.calib_logodds_eps,
                                     group_uuid=group_uuid, group_name=group_name
                                     )

                    oframe2 = _get_new_col(icol.name + "_inverse")
                    mojo += MjT_ConstBinaryOp(iframe=oframe1, oframe=oframe2, op="subtract", const=1., pos="left",
                                              group_uuid=group_uuid, group_name=group_name)

                    oframe3 = _get_new_col(icol.name + "_ratio")
                    pair = _get_new_pair(oframe1, oframe2)
                    mojo += MjT_BinaryOp(iframe=pair, oframe=oframe3, op="divide", eps=1e-10, group_uuid=group_uuid,
                                         group_name=group_name)

                    oframe4 = _get_new_col(icol.name + "_log")
                    mojo += MjT_Log(iframe=oframe3, oframe=oframe4, group_uuid=group_uuid, group_name=group_name)

                    inp = oframe4
                else:
                    inp = _iframe[c]

                knots = self.calib_knot_vec_tr[c]
                num_knots = len(knots)

                # zero col
                zeros = _get_new_col(icol.name + "_zeros")
                mojo += MjT_ConstBinaryOp(iframe=inp, oframe=zeros, op="multiply", const=0., pos="right",
                                          group_uuid=group_uuid, group_name=group_name)

                # ones col
                ones = _get_new_col(icol.name + f"_ones")
                mojo += MjT_ConstBinaryOp(iframe=zeros, oframe=ones, op="add", const=1., pos="right",
                                          group_uuid=group_uuid, group_name=group_name)

                # last knot calc
                denom = knots[-1] - knots[-2]

                def _to_mojo_helper(mojo, inp, val, zeros, suffix=""):
                    oframe5 = _get_new_col(icol.name + f"_{suffix}diff")
                    mojo += MjT_ConstBinaryOp(iframe=inp, oframe=oframe5, op="subtract", const=val, pos="right",
                                              group_uuid=group_uuid, group_name=group_name)

                    oframe6 = _get_new_col(icol.name + f"_{suffix}max")
                    pair = _get_new_pair(oframe5, zeros)
                    mojo += MjT_Agg(iframe=pair, oframe=oframe6, op="max", group_uuid=group_uuid, group_name=group_name)

                    oframe7 = _get_new_col(icol.name + f"_{suffix}pwr")
                    oframe8 = _get_new_col(icol.name + f"_{suffix}pwr2")
                    oframe9 = _get_new_col(icol.name + f"_{suffix}pwr3")
                    mojo += MjT_ConstBinaryOp(iframe=oframe6, oframe=oframe7, op="multiply", const=1., pos="right",
                                              group_uuid=group_uuid, group_name=group_name)
                    pair = _get_new_pair(oframe6, oframe7)
                    mojo += MjT_BinaryOp(iframe=pair, oframe=oframe8, op="multiply", group_uuid=group_uuid,
                                         group_name=group_name)
                    pair = _get_new_pair(oframe8, oframe7)
                    mojo += MjT_BinaryOp(iframe=pair, oframe=oframe9, op="multiply", group_uuid=group_uuid,
                                         group_name=group_name)
                    return oframe9

                last_knot2 = _to_mojo_helper(mojo=mojo, inp=inp, val=knots[-2], zeros=zeros, suffix="last2")
                last_knot1 = _to_mojo_helper(mojo=mojo, inp=inp, val=knots[-1], zeros=zeros, suffix="last1")

                oframe5 = _get_new_col(icol.name + f"_last21diff")
                pair = _get_new_pair(last_knot2, last_knot1)
                mojo += MjT_BinaryOp(iframe=pair, oframe=oframe5, op="subtract", group_uuid=group_uuid,
                                     group_name=group_name)

                last_knot = _get_new_col(icol.name + f"_lastKnot")
                mojo += MjT_ConstBinaryOp(iframe=oframe5, oframe=last_knot, op="divide", const=denom, pos="right",
                                          group_uuid=group_uuid, group_name=group_name)

                # all knots calc
                results = []

                for i in range(1, num_knots - 1):
                    denom = knots[-1] - knots[i - 1]

                    knot1 = _to_mojo_helper(mojo=mojo, inp=inp, val=knots[i - 1], zeros=zeros, suffix=f"knot{i}m1")
                    knot2 = _to_mojo_helper(mojo=mojo, inp=inp, val=knots[-1], zeros=zeros, suffix=f"knotm1f{i}")

                    oframe_ = _get_new_col(icol.name + f"_knots_{i}_diff")
                    pair = _get_new_pair(knot1, knot2)
                    mojo += MjT_BinaryOp(iframe=pair, oframe=oframe_, op="subtract", group_uuid=group_uuid,
                                         group_name=group_name)

                    div_res = _get_new_col(icol.name + f"_dv_{i}")
                    mojo += MjT_ConstBinaryOp(iframe=oframe_, oframe=div_res, op="divide", const=denom, pos="right",
                                              group_uuid=group_uuid, group_name=group_name)

                    diff_res = _get_new_col(icol.name + f"_diff_{i}")
                    pair = _get_new_pair(div_res, last_knot)
                    mojo += MjT_BinaryOp(iframe=pair, oframe=diff_res, op="subtract", group_uuid=group_uuid,
                                         group_name=group_name)
                    results.append(diff_res)

                results = [ones, inp] + results

                assert len(results) == len(self.calib_basis_coef_vec[c].ravel()), "Something went wrong :("
                # linear model
                results2 = MojoFrame()
                for i, (frame_, const_) in enumerate(zip(results, self.calib_basis_coef_vec[c].ravel())):
                    res_fr = _get_new_col(icol.name + f"_logits_{i}")
                    mojo += MjT_ConstBinaryOp(iframe=frame_, oframe=res_fr, op="multiply", const=const_, pos="right",
                                              group_uuid=group_uuid, group_name=group_name)
                    results2.cbind(res_fr)

                ocol_logits_sum = _get_new_col(icol.name + f"_logits_sum")
                mojo += MjT_Agg(iframe=results2, oframe=ocol_logits_sum, op="sum", group_uuid=group_uuid,
                                group_name=group_name)

                # sigmoid
                ocol_spline_sigmoid = _get_new_col(icol.name + f"_spline_sigmoid", type_="float64")
                mojo += MjT_Sigmoid(iframe=ocol_logits_sum, oframe=ocol_spline_sigmoid, group_uuid=group_uuid,
                                    group_name=group_name)
                ocol_spline_sigmoid_astype = _get_new_col(icol.name + f"_spline_sigmoid_astype")
                mojo += MjT_AsType(
                    iframe=ocol_spline_sigmoid, oframe=ocol_spline_sigmoid_astype,
                    type="float32",
                    group_uuid=group_uuid, group_name=group_name
                )
                res.cbind(ocol_spline_sigmoid_astype)

            else:
                raise RuntimeError('Unknown calibration method in to_mojo()')
        # normalization
        if len(res) > 1:
            res2 = MojoFrame()
            oframe_sum = _get_new_col(self.__class__.__name__ + "_sum")
            mojo += MjT_Agg(iframe=res, oframe=oframe_sum, op="sum", group_uuid=group_uuid, group_name=group_name)

            for c in range(len(res)):
                icol = res.get_column(c)
                oframe1 = _get_new_col(icol.name + "_normalized")

                pair = _get_new_pair(res[c], oframe_sum)
                mojo += MjT_BinaryOp(iframe=pair, oframe=oframe1, op="divide", group_uuid=group_uuid,
                                     group_name=group_name)
                res2.cbind(oframe1)

            res = res2

        return res
