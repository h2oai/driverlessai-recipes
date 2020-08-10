""" Calibrated Classifier Model: To calibrate predictions using Platt's scaling
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

class CalibratedClassifierModel:
    _regression = False
    _binary = True
    _multiclass = True
    _can_use_gpu = True
    _mojo = True
    _description = "Calibrated Classifier Model (LightGBM)"

    le = LabelEncoder()

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
        if eval_set is not None:
            eval_set_y = self.le.transform(eval_set[0][1])
            eval_set_classification = [(eval_set[0][0], eval_set_y.astype(int))]
            
        #Stratified split with classes control - making sure all classes present in both train and test
        unique_cls = np.unique(y_)
        tr_indx, te_indx = [], []

        for c in unique_cls:
            c_indx = np.argwhere(y_==c).ravel()
            indx = np.random.permutation(c_indx)
            start_indx = max(1, int(self.params["calib_perc"]*len(c_indx))) # at least 1 exemplar should be presented
            tr_indx += list(indx[start_indx:])
            te_indx += list(indx[:start_indx])
        tr_indx = np.array(tr_indx)
        te_indx = np.array(te_indx)

        X_train, y_train = X[tr_indx, :], y_.astype(int)[tr_indx]
        X_calibrate, y_calibrate = X[te_indx, :].to_pandas(), np.array(y)[te_indx].ravel()

        if sample_weight is not None:
            sample_weight_ = sample_weight[tr_indx]
            sample_weight_calib = sample_weight[te_indx]
        else:
            sample_weight_ = sample_weight
            sample_weight_calib = sample_weight

        model_classification.fit(X_train, y_train,
                                 sample_weight=sample_weight_, eval_set=eval_set_classification,
                                 sample_weight_eval_set=sample_weight_eval_set, **kwargs)

        # calibration
        model_classification.predict_proba = model_classification.predict_simple
        model_classification.classes_ = self.le.classes_
        calibrator = CalibratedClassifierCV(
            base_estimator=model_classification,
            method=self.params["calib_method"],
            cv='prefit')

        calibrator.fit(X_calibrate, y_calibrate, sample_weight = sample_weight_calib)
        
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
                self._necessary_X_.append(c._necessary_X_)
                self._necessary_y_.append(c._necessary_y_)
                
                self.X_min_.append(c.X_min_)
                self.X_max_.append(c.X_max_)
                
        
        else:
            raise RuntimeError('Unknown calibration method in fit()')
        # calibration

        varimp = model_classification.imp_features(columns=X.names)
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

    def predict(self, X, **kwargs):
        from scipy import interpolate
        
        X = dt.Frame(X)
        model, _, _, _ = self.get_model_properties()
        preds = model.predict_proba(X)
        
        if preds.shape[1] <= 2:
            if self.calib_method == "sigmoid":
                scaled_preds = expit(-(self.slope[0] * preds[:,1] + self.intercept[0]))
            elif self.calib_method == "isotonic":
                f_ = interpolate.interp1d(
                    self._necessary_X_[0], 
                    self._necessary_y_[0], 
                    kind='linear',
                    bounds_error='nan'
                )
                scaled_preds = f_(np.clip(preds[:,1], self.X_min_[0], self.X_max_[0]))
            else:
                raise RuntimeError('Unknown calibration method in predict()')
            preds[:,1] = scaled_preds
            preds[:,0] = 1. - scaled_preds
        else:
            for c in range(preds.shape[1]):
                if self.calib_method == "sigmoid":
                    scaled_preds = expit(-(self.slope[c] * preds[:,c] + self.intercept[c]))
                elif self.calib_method == "isotonic":
                    f_ = interpolate.interp1d(
                        self._necessary_X_[c], 
                        self._necessary_y_[c], 
                        kind='linear',
                        bounds_error='nan'
                    )
                    scaled_preds = f_(np.clip(preds[:,c], self.X_min_[c], self.X_max_[c]))
                else:
                    raise RuntimeError('Unknown calibration method in predict()')
    
                preds[:,c] = scaled_preds
            preds = preds / np.sum(preds, 1).reshape(-1,1)
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
        self.params["calib_perc"] = .1
        self.params["calib_method"] = "sigmoid"

    def mutate_params(self, **kwargs):
        super().mutate_params(**kwargs)
        self.params["calib_perc"] = np.random.choice([.05, .1, .15, .2])
        self.params["calib_method"] =  np.random.choice(["isotonic", "sigmoid"])
    
    def write_to_mojo(self, mojo: MojoWriter, iframe: MojoFrame, group_uuid=None, group_name=None):
        return self.to_mojo(mojo = mojo, iframe = iframe, group_uuid=group_uuid, group_name=group_name)
    
    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame, group_uuid=None, group_name=None):
        
        from h2oaicore.mojo import MojoColumn
        from h2oaicore.mojo_transformers import MjT_ConstBinaryOp, MjT_Sigmoid, MjT_AsType, MjT_Agg, MjT_BinaryOp, MjT_IntervalMap, MjT_Clip
        import uuid
        group_uuid = str(uuid.uuid4())
        group_name = self.__class__.__name__

        
        _iframe = super().write_to_mojo(mojo = mojo, iframe = iframe, group_uuid=group_uuid, group_name=group_name)
        res = MojoFrame()
        
        for c in range(len(_iframe)):
            icol = _iframe.get_column(c)
            
            if self.calib_method == "sigmoid":
                
                ocol1 = MojoColumn(name=icol.name + "_slope", dtype=icol.type)
                oframe1 = MojoFrame(columns=[ocol1])
                ocol2 = MojoColumn(name=icol.name + "_intercept", dtype=icol.type)
                oframe2 = MojoFrame(columns=[ocol2])
                ocol3 = MojoColumn(name=icol.name + "_negative", dtype=icol.type)
                oframe3 = MojoFrame(columns=[ocol3])
                ocol4 = MojoColumn(name=icol.name + "_calibrated", dtype="float64")
                oframe4 = MojoFrame(columns=[ocol4])
                ocol5 = MojoColumn(name=icol.name + "_astype", dtype=icol.type)
                oframe5 = MojoFrame(columns=[ocol5])


                mojo += MjT_ConstBinaryOp(iframe=_iframe[c], oframe=oframe1, op="multiply", const= self.slope[c], pos="right",
                                          group_uuid=group_uuid, group_name=group_name)

                mojo += MjT_ConstBinaryOp(iframe=oframe1, oframe=oframe2, op="add", const=self.intercept[c], pos="right",
                                          group_uuid=group_uuid, group_name=group_name)

                mojo += MjT_ConstBinaryOp(iframe=oframe2, oframe=oframe3, op="multiply", const=-1., pos="right",
                                          group_uuid=group_uuid, group_name=group_name)

                mojo+= MjT_Sigmoid(iframe=oframe3, oframe=oframe4, group_uuid=group_uuid, group_name=group_name)
                mojo+= MjT_AsType(iframe=oframe4, oframe=oframe5, type = "float32", group_uuid=group_uuid, group_name=group_name)


                res.cbind(oframe5)
            elif self.calib_method == "isotonic":
                X = list(self._necessary_X_[c])
                y = list(self._necessary_y_[c])
                if len(y) == 1:
                    new_y = _iframe[c]
                    ocol1 = MojoColumn(name=icol.name + "_zeroing", dtype=icol.type)
                    oframe1 = MojoFrame(columns=[ocol1])
                    ocol2 = MojoColumn(name=icol.name + "_addingConst", dtype=icol.type)
                    new_y = MojoFrame(columns=[ocol2])
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

                    XY = MojoFrame(columns=[ocol1,ocol2,ocol3, ocol4])
                    
                    #clipping
                    ocol_cl = MojoColumn(name=icol.name + "_clipped", dtype=icol.type)
                    inp_clipped = MojoFrame(columns=[ocol_cl])
                    mojo+= MjT_Clip(iframe=_iframe[c], oframe=inp_clipped, 
                                    min = self.X_min_[c], max = self.X_max_[c],
                                    group_uuid=group_uuid, group_name=group_name
                                   )

                    #search for coordinates
                    mojo+= MjT_IntervalMap(
                        iframe=inp_clipped, oframe=XY, 
                        breakpoints = X, 
                        values = [[x1, x0, y1, y0] for x1, x0, y1, y0 in zip(max_X, min_X, max_y, min_y)], 
                        group_uuid=group_uuid, group_name=group_name
                    )

                    #interpolation
                    ocol5 = MojoColumn(name=icol.name + "_currDiff", dtype=icol.type)
                    curr_diff = MojoFrame(columns=[ocol5])
                    pair = MojoFrame()
                    pair.cbind(inp_clipped)
                    pair.cbind(XY[1])
                    mojo+= MjT_BinaryOp(iframe = pair, oframe=curr_diff, op = "subtract", group_uuid=group_uuid, group_name=group_name)

                    ocol6 = MojoColumn(name=icol.name + "_yDiff", dtype=icol.type)
                    y_diff = MojoFrame(columns=[ocol6])
                    pair = MojoFrame()
                    pair.cbind(XY[2])
                    pair.cbind(XY[3])
                    mojo+= MjT_BinaryOp(iframe = pair, oframe=y_diff, op = "subtract", group_uuid=group_uuid, group_name=group_name)

                    ocol7 = MojoColumn(name=icol.name + "_XDiff", dtype=icol.type)
                    X_diff = MojoFrame(columns=[ocol7])
                    pair = MojoFrame()
                    pair.cbind(XY[0])
                    pair.cbind(XY[1])
                    mojo+= MjT_BinaryOp(iframe = pair, oframe=X_diff, op = "subtract", group_uuid=group_uuid, group_name=group_name)

                    ocol8 = MojoColumn(name=icol.name + "_xyRatio", dtype=icol.type)
                    xy_ratio = MojoFrame(columns=[ocol8])
                    pair = MojoFrame()
                    pair.cbind(y_diff)
                    pair.cbind(X_diff)
                    mojo+= MjT_BinaryOp(iframe = pair, oframe=xy_ratio, op = "divide", eps = 1e-10, group_uuid=group_uuid, group_name=group_name)

                    ocol9 = MojoColumn(name=icol.name + "_scaledCurDiff", dtype=icol.type)
                    scaled_cur_diff = MojoFrame(columns=[ocol9])
                    pair = MojoFrame()
                    pair.cbind(xy_ratio)
                    pair.cbind(curr_diff)
                    mojo+= MjT_BinaryOp(iframe = pair, oframe=scaled_cur_diff, op = "multiply", group_uuid=group_uuid, group_name=group_name)

                    ocol10 = MojoColumn(name=icol.name + "_newY", dtype=icol.type)
                    new_y = MojoFrame(columns=[ocol10])
                    pair = MojoFrame()
                    pair.cbind(XY[3])
                    pair.cbind(scaled_cur_diff)
                    mojo+= MjT_BinaryOp(iframe = pair, oframe=new_y, op = "add", group_uuid=group_uuid, group_name=group_name)
                
                res.cbind(new_y)
            else:
                raise RuntimeError('Unknown calibration method in to_mojo()')
            
        if len(res) > 1:
            res2 = MojoFrame()
            ocol_sum = MojoColumn(name=self.__class__.__name__ + "_sum", dtype="float32")
            oframe_sum = MojoFrame(columns=[ocol_sum])
            mojo+= MjT_Agg(iframe=res, oframe=oframe_sum, op = "sum", group_uuid=group_uuid, group_name=group_name)

            for c in range(len(res)):
                icol = res.get_column(c)
                ocol1 = MojoColumn(name=icol.name + "_normalized", dtype=icol.type)
                oframe1 = MojoFrame(columns=[ocol1])

                pair = MojoFrame()
                pair.cbind(res[c])
                pair.cbind(oframe_sum)

                mojo+= MjT_BinaryOp(iframe = pair, oframe=oframe1, op = "divide", group_uuid=group_uuid, group_name=group_name)
                res2.cbind(oframe1)

            res = res2

        return res
