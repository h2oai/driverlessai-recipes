import numpy as np
import pandas as pd
import torch
from datatable import dt
from h2oaicore.models import CustomTimeSeriesTensorFlowModel
from sklearn.preprocessing import StandardScaler

# The recipe depends on Time knob values: it uses subset of training data for lower Time values to speed things up.
# It is recommended to start with Time=1 and try increasing it if runtime allows it given the dataset size.

class DataPrep:
    def __init__(self, config):
        config["tgc"] = list(set(config["tgc"]) - set([config["time_column"]]))
        self.config = config

    def fit(self, df):

        self.config["tgc_nval"] = []
        if len(self.config["tgc"]) > 0:
            self.tgc_maps = {}
            self.tgc_topv = {}
            for tgc in self.config["tgc"]:
                vc = df[tgc].value_counts()
                self.config["tgc_nval"].append(len(vc))
                self.tgc_topv[tgc] = vc.head(1).index[0]
                self.tgc_maps[tgc] = {
                    v: i for i, v in enumerate(df[tgc].value_counts().index)
                }

        self.config["knatt"] = list(
            set(df.columns)
            - set(self.config["tgc"])
            - set([self.config["time_column"]])
            - set([self.config["target_column"]])
            - set(self.config["unatt"])
            - set(self.config["drop_columns"])
        )
        if len(self.config["knatt"]) > 0:
            self.knatt_scaler = StandardScaler().fit(df[self.config["knatt"]])
        if len(self.config["unatt"]) > 0:
            self.unatt_scaler = StandardScaler().fit(df[self.config["unatt"]])
        self.config["X0_input_features"] = (
            df[self.config["knatt"]].shape[1] + df[self.config["unatt"]].shape[1]
        )
        self.config["X1_input_features"] = df[self.config["knatt"]].shape[1]
        self.config["y_mean"] = df[self.config["target_column"]].mean()
        self.config["y_std"] = df[self.config["target_column"]].std()

        return self

    def transform(self, df, data_type="train"):
        res = {"df": df}

        res["X"] = np.zeros(len(df)).reshape(-1, 1)

        if len(self.config["knatt"]) > 0:
            res["X"] = np.hstack(
                [res["X"], self.knatt_scaler.transform(df[self.config["knatt"]])]
            )
        if (len(self.config["unatt"]) > 0) and (data_type == "train"):
            res["X"] = np.hstack(
                [res["X"], self.unatt_scaler.transform(df[self.config["unatt"]])]
            )
        if data_type == "train":
            if self.config["num_classes"] == 1:
                res["y"] = (
                    df[self.config["target_column"]].values - self.config["y_mean"]
                ) / self.config["y_std"]
            else:
                res["y"] = df[self.config["target_column"]].values

        if res["X"].shape[1] > 1:
            res["X"] = res["X"][:, 1:]

        for i in range(res["X"].shape[1]):
            res["X"][np.isnan(res["X"][:, i]), i] = np.nanmean(res["X"][:, i])

        res["tgc"] = []
        for tgc in self.config["tgc"]:
            res["tgc"].append(
                df[tgc]
                .map(self.tgc_maps[tgc])
                .fillna(self.tgc_topv[tgc])
                .astype(int)
                .values.reshape(-1, 1)
            )
        if len(res["tgc"]) > 0:
            res["tgc"] = np.hstack(res["tgc"])
        return res


class RNNDS_train(torch.utils.data.Dataset):
    def __init__(self, config, df, dprep):
        self.config = config
        self.df = df.copy()

        time_vals_unique = np.sort(self.df[config["time_column"]].unique())
        self.max_seq_len = len(time_vals_unique) - config["forecast_horizon"]
        max_fit_time = time_vals_unique[-config["forecast_horizon"]]

        if len(config["tgc"]) > 0:
            self.fit_records_mask = (
                self.df.groupby(config["tgc"]).cumcount() >= config["min_seq_len"]
            ) & (
                self.df.iloc[::-1].groupby(config["tgc"]).cumcount().iloc[::-1]
                >= config["forecast_horizon"]
            )
        else:
            self.fit_records_mask = (self.df.index >= config["min_seq_len"]) & (
                self.df[config["time_column"]] < max_fit_time
            )

        d = dprep.transform(df, "train")
        self.X0 = d["X"]
        self.tgc = d["tgc"]
        self.y = d["y"]
        d = dprep.transform(df, "test")
        self.X1 = d["X"]
        self.idxs = np.arange(self.fit_records_mask.sum())
        np.random.shuffle(self.idxs)

    def __len__(self):
        if self.config["time_setting"] < 6:
            return len(self.idxs) // (2 ** (6 - self.config["time_setting"]))
        else:
            return len(self.idxs)

    def __getitem__(self, idx0):
        idx = self.idxs[idx0]
        rec = self.df[self.fit_records_mask].iloc[idx]
        mask0 = self.df[self.config["time_column"]] <= rec[self.config["time_column"]]
        mask1 = self.df[self.config["time_column"]] > rec[self.config["time_column"]]
        for col in self.config["tgc"]:
            mask0 = mask0 & (self.df[col] == rec[col])
            mask1 = mask1 & (self.df[col] == rec[col])
        X0 = self.X0[mask0]
        y0 = self.y[mask0]
        X1 = self.X1[mask1]
        y1 = self.y[mask1]
        if len(self.tgc) > 0:
            tgc0 = self.tgc[mask0]
            tgc1 = self.tgc[mask1]
        else:
            tgc0 = []
            tgc1 = []
        if len(y1) >= self.config["forecast_horizon"]:
            X1 = X1[: self.config["forecast_horizon"]]
            y1 = y1[: self.config["forecast_horizon"]]
            tgc1 = tgc1[: self.config["forecast_horizon"]]
        return {"X0": X0, "y0": y0, "tgc0": tgc0, "X1": X1, "y1": y1, "tgc1": tgc1}


class RNNDS_test(torch.utils.data.Dataset):
    def __init__(self, config, train_df, test_df, dprep):
        self.config = config
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()

        self.test_dates = list(self.test_df[config["time_column"]].unique())

        d = dprep.transform(train_df, "train")
        self.X0 = d["X"]
        self.tgc0 = d["tgc"]
        self.y = d["y"]
        d = dprep.transform(test_df, "test")
        self.X1 = d["X"]
        self.tgc1 = d["tgc"]

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx):
        rec = self.test_df.iloc[idx]
        mask0 = [True] * len(self.X0)
        mask1 = [True] * len(self.X1)
        for col in self.config["tgc"]:
            mask0 = mask0 & (self.train_df[col] == rec[col])
            mask1 = mask1 & (self.test_df[col] == rec[col])
        X0 = self.X0[mask0]
        y0 = self.y[mask0]
        X1 = self.X1[mask1]
        if len(self.tgc0) > 0:
            tgc0 = self.tgc0[mask0]
            tgc1 = self.tgc1[mask1]
        else:
            tgc0 = []
            tgc1 = []
        return {
            "X0": X0,
            "y0": y0,
            "tgc0": tgc0,
            "X1": X1,
            "tgc1": tgc1,
            # "seqlen1": self.test_dates.index(rec[self.config["time_column"]]),
            "seqlen1": min(
                self.config["forecast_horizon"] - 1,
                self.test_dates.index(rec[self.config["time_column"]]),
            ),
        }


class SingleRNN(torch.nn.Module):
    def __init__(self, config, dilation=1):
        super(SingleRNN, self).__init__()
        self.config = config
        self.dilation = dilation

        if config["rnn_type"] == "gru":
            rnn_class = torch.nn.GRU
        elif config["rnn_type"] == "lstm":
            rnn_class = torch.nn.LSTM
        elif config["rnn_type"] == "elman":
            rnn_class = torch.nn.RNN

        self.rnn_encoder = rnn_class(
            input_size=config["m_rnn_units"] * 2,
            hidden_size=config["m_rnn_units"],
            num_layers=config["m_encoder_num_layers"],
            bidirectional=False,
            dropout=config["m_encoder_dropout"],
            batch_first=True,
        )

        self.decoder = rnn_class(
            input_size=config["m_rnn_units"] * 2,
            hidden_size=config["m_rnn_units"],
            num_layers=config["m_decoder_num_layers"],
            bidirectional=False,
            dropout=config["m_decoder_dropout"],
            batch_first=True,
        )

    def forward(self, X0, X1, seqlen):

        bsize = X0.shape[0]
        tgt_len = self.config["forecast_horizon"] // self.dilation + 1

        X0 = X0[:, :, -self.dilation * (X0.shape[-1] // self.dilation) :]  # trim seqlen
        XX0 = torch.stack([X0[:, :, i :: self.dilation] for i in range(self.dilation)])
        XX0 = XX0.reshape(
            bsize * self.dilation, XX0.shape[2], -1
        )  # batch*dilation | 2*emb | seqlen/dilation

        if self.config["rnn_type"] in ["gru", "elman"]:
            rnn_out, rnn_hidden = self.rnn_encoder(
                XX0.transpose(1, 2)
            )  # batch*dilation | seqlen/dilation | emb
        elif self.config["rnn_type"] in ["lstm"]:
            rnn_out, (rnn_hidden, rnn_cstate) = self.rnn_encoder(
                XX0.transpose(1, 2)
            )  # batch*dilation | seqlen/dilation | emb

        X1 = torch.nn.functional.pad(
            X1, (0, tgt_len * self.dilation - X1.shape[-1], 0, 0, 0, 0)
        )
        X1 = torch.stack([X1[:, :, i :: self.dilation] for i in range(self.dilation)])
        X1 = X1.reshape(
            bsize * self.dilation, X1.shape[2], -1
        )  # batch*dilation | emb | seqlen/dilation

        outputs = torch.zeros(
            tgt_len,
            bsize * self.dilation,
            rnn_hidden.shape[-1],
            device=self.config["device"],
        )
        decoder_hidden = rnn_hidden
        decoder_input = rnn_out[:, -1].unsqueeze(1)
        for t in range(tgt_len):
            decoder_input = torch.cat(
                [decoder_input, X1[:, :, t].unsqueeze(1)], -1
            )  # batch*dilation | 1 | 2*emb
            if self.config["rnn_type"] in ["gru", "elman"]:
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden
                )
            elif self.config["rnn_type"] in ["lstm"]:
                decoder_output, (decoder_hidden, rnn_cstate) = self.decoder(
                    decoder_input, (decoder_hidden, rnn_cstate)
                )
            outputs[t] = decoder_output.squeeze(1)
            decoder_input = decoder_output

        outputs = outputs.permute(1, 2, 0)
        outputs = (
            outputs.reshape(self.dilation, bsize, outputs.shape[1], -1)
            .permute(1, 2, 3, 0)
            .reshape(bsize, outputs.shape[1], -1)
        )
        return outputs[:, :, : self.config["forecast_horizon"]]


class TSRNN(torch.nn.Module):
    def __init__(self, config):
        super(TSRNN, self).__init__()
        self.config = config

        self.encoder_X0 = torch.nn.Sequential(
            torch.nn.Conv1d(
                config["X0_input_features"]
                + len(config["tgc_nval"]) * config["m_emb_dim"],
                config["m_rnn_units"],
                1,
                bias=True,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(config["m_rnn_units"], config["m_rnn_units"], 1, bias=True),
        )
        self.encoder_X1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                config["X1_input_features"]
                + len(config["tgc_nval"]) * config["m_emb_dim"],
                config["m_rnn_units"],
                1,
                bias=True,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(config["m_rnn_units"], config["m_rnn_units"], 1, bias=True),
        )
        self.encoder_y = torch.nn.Sequential(
            torch.nn.Conv1d(1, config["m_rnn_units"], 1, bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(config["m_rnn_units"], config["m_rnn_units"], 1, bias=True),
        )

        self.rnns = torch.nn.ModuleList(
            [SingleRNN(config, dil) for dil in config["dilations"]]
        )
        self.embs = torch.nn.ModuleList(
            [
                torch.nn.Embedding(nval, config["m_emb_dim"])
                for nval in config["tgc_nval"]
            ]
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv1d(
                len(config["dilations"]) * config["m_rnn_units"],
                config["m_last_linear_units"],
                1,
                bias=True,
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                config["m_last_linear_units"],
                self.config["num_classes"] if self.config["num_classes"] > 2 else 1,
                1,
                bias=True,
            ),
        )

    def forward(self, d):
        if len(self.config["tgc_nval"]) > 0:
            tgc0 = torch.cat(
                [emb(d["tgc0"][:, :, i]) for i, emb in enumerate(self.embs)], -1
            ).transpose(1, 2)
            if self.config["X0_input_features"] > 0:
                X0 = torch.cat([d["X0"], tgc0], 1)
            else:
                X0 = tgc0
        else:
            X0 = d["X0"]
        X0 = torch.cat(
            [self.encoder_X0(X0), self.encoder_y(d["y0"].unsqueeze(1))], 1
        )  # batch | 2*emb | seqlen

        if len(self.config["tgc_nval"]) > 0:
            tgc1 = torch.cat(
                [emb(d["tgc1"][:, :, i]) for i, emb in enumerate(self.embs)], -1
            ).transpose(1, 2)
            if self.config["X1_input_features"] > 0:
                X1 = torch.cat([d["X1"], tgc1], 1)
            else:
                X1 = tgc1
        else:
            X1 = d["X1"]
        X1 = self.encoder_X1(X1)  # batch |   emb | seqlen

        dec = torch.cat([rnn(X0, X1, d["seqlen0"]) for rnn in self.rnns], 1)
        return self.decoder(dec)


def collator(batch):
    seqlen0 = torch.stack([torch.tensor(len(b["X0"])) for b in batch])
    if "seqlen1" in batch[0].keys():
        seqlen1 = torch.stack([torch.tensor(b["seqlen1"]) for b in batch])
    else:
        seqlen1 = torch.tensor([])
    max0 = torch.max(seqlen0)
    max1 = torch.max(torch.tensor([len(b["X1"]) for b in batch]))
    X0 = torch.stack(
        [
            torch.nn.functional.pad(
                torch.tensor(b["X0"]),
                (0, 0, 0, max0 - len(b["X0"])),
                mode="constant",
                value=0,
            )
            for b in batch
        ]
    )
    if len(batch[0]["tgc0"]) > 0:
        tgc0 = torch.stack(
            [
                torch.nn.functional.pad(
                    torch.tensor(b["tgc0"]),
                    (0, 0, 0, max0 - len(b["tgc0"])),
                    mode="constant",
                    value=0,
                )
                for b in batch
            ]
        )
    else:
        tgc0 = torch.tensor([])
    X1 = torch.stack(
        [
            torch.nn.functional.pad(
                torch.tensor(b["X1"]),
                (0, 0, 0, max1 - len(b["X1"])),
                mode="constant",
                value=0,
            )
            for b in batch
        ]
    )
    if len(batch[0]["tgc1"]) > 0:
        tgc1 = torch.stack(
            [
                torch.nn.functional.pad(
                    torch.tensor(b["tgc1"]),
                    (0, 0, 0, max1 - len(b["tgc1"])),
                    mode="constant",
                    value=0,
                )
                for b in batch
            ]
        )
    else:
        tgc1 = torch.tensor([])
    y0 = torch.stack(
        [
            torch.nn.functional.pad(
                torch.tensor(b["y0"]),
                (0, max0 - len(b["y0"])),
                mode="constant",
                value=0,
            )
            for b in batch
        ]
    )
    if "y1" in batch[0].keys():
        y1 = torch.stack(
            [
                torch.nn.functional.pad(
                    torch.tensor(b["y1"]),
                    (0, max1 - len(b["y1"])),
                    mode="constant",
                    value=0,
                )
                for b in batch
            ]
        )
    else:
        y1 = torch.tensor([])
    return {
        "X0": X0,
        "X1": X1,
        "y0": y0,
        "y1": y1,
        "tgc0": tgc0,
        "tgc1": tgc1,
        "seqlen0": seqlen0,
        "seqlen1": seqlen1,
    }


def fit(model, loader, config):
    model.train()
    model.to(config["device"])
    loss_function = config["loss_function"]()
    optimizer = config["optimizer"](model.parameters(), config["lr"])
    if config["lr_scheduler"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config["n_epochs"], eta_min=1e-7
        )
    for epoch in range(config["n_epochs"]):
        optimizer.zero_grad()
        for i, d in enumerate(loader):
            d["X0"] = d["X0"].transpose(1, 2).to(config["device"]).float()
            d["X1"] = d["X1"].transpose(1, 2).to(config["device"]).float()
            d["y0"] = d["y0"].to(config["device"]).float()
            if config["num_classes"] > 2:
                d["y1"] = d["y1"].to(config["device"]).long()
            else:
                d["y1"] = d["y1"].to(config["device"]).float()
            d["tgc0"] = d["tgc0"].to(config["device"])
            d["tgc1"] = d["tgc1"].to(config["device"])
            d["seqlen0"] = d["seqlen0"].to(config["device"])
            d["seqlen1"] = d["seqlen1"].to(config["device"])

            preds = model(d).transpose(1, 2)
            if config["num_classes"] > 2:
                loss = loss_function(
                    preds.reshape(preds.shape[0] * preds.shape[1], preds.shape[2]),
                    d["y1"].reshape(d["y1"].shape[0] * d["y1"].shape[1]),
                )
            elif config["num_classes"] == 2:
                loss = loss_function(
                    torch.sigmoid(preds).reshape(preds.shape[0] * preds.shape[1]),
                    d["y1"].reshape(d["y1"].shape[0] * d["y1"].shape[1]),
                )
            else:
                loss = loss_function(d["y1"], preds.squeeze(2))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if config["lr_scheduler"]:
                scheduler.step()
    return model


def predict(model, loader, config):
    model.eval()
    model.to(config["device"])
    preds = []
    for i, d in enumerate(loader):
        d["X0"] = d["X0"].transpose(1, 2).to(config["device"]).float()
        d["X1"] = d["X1"].transpose(1, 2).to(config["device"]).float()
        d["y0"] = d["y0"].to(config["device"]).float()
        d["tgc0"] = d["tgc0"].to(config["device"])
        d["tgc1"] = d["tgc1"].to(config["device"])
        d["seqlen0"] = d["seqlen0"].to(config["device"])
        # d['seqlen1'] = d['seqlen1'].to(config['device'])

        if config["num_classes"] == 1:
            preds_delta = (
                model(d).squeeze(1).detach().cpu().numpy() * config["y_std"]
                + config["y_mean"]
            )
        elif config["num_classes"] == 2:
            preds_delta = torch.sigmoid(model(d)).transpose(1, 2).detach().cpu().numpy()
        else:
            preds_delta = model(d).transpose(1, 2).detach().cpu().numpy()
        preds.append([preds_delta[i, d["seqlen1"][i]] for i in range(len(preds_delta))])

    return np.concatenate(preds)


class TS_RNN(CustomTimeSeriesTensorFlowModel):
    _display_name = "TS_RNN"
    _description = "RNN for time series"
    _regression = True  # y has shape (N,) and is of numeric type, no missing values
    _binary = True  # y has shape (N,) and can be numeric or string, cardinality 2, no missing values
    _multiclass = True  # y has shape (N,) and can be numeric or string, cardinality 3+, no missing values
    _can_handle_categorical = False
    _can_use_gpu = True
    _is_reproducible = False
    _parallel_task = True

    config = dict(
        # device=["cpu", "cuda"][0],
        num_workers=8,
        optimizer=torch.optim.Adam,
        m_rnn_units=32,
        m_emb_dim=16,
        m_encoder_num_layers=1,
        m_decoder_num_layers=1,
        m_encoder_dropout=0,
        m_decoder_dropout=0,
        m_last_linear_units=32,
    )

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def do_acceptance_test():
        return True

    def set_default_params(
        self, accuracy=10, time_tolerance=10, interpretability=1, **kwargs
    ):
        self.mutate_params(accuracy, time_tolerance, interpretability)

    def mutate_params(
        self, accuracy=10, time_tolerance=10, interpretability=1, **kwargs
    ):
        # self.config["n_epochs"] = np.random.choice([5])
        if time_tolerance > 6:
            self.config["n_epochs"] = 5 + time_tolerance - 6
        else:
            self.config["n_epochs"] = 5
        self.config["batch_size"] = int(np.random.choice([16, 32]))
        self.config["rnn_type"] = np.random.choice(["gru", "lstm", "elman"])
        self.config["lr"] = np.random.choice([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
        self.config["lr_scheduler"] = np.random.choice([True, False])
        self.config["dilations"] = [1] + list(
            np.random.choice(
                [3, 4, 5, 7, 12, 28, 52, 365],
                np.random.choice([0, 1, 2]),
                replace=False,
            )
        )
        self.config["time_setting"] = time_tolerance
        self.config["device"] = (
            "cuda" if (self.params_base.get("n_gpus", 0) > 0) else "cpu"
        )

    def fit(
        self,
        X: dt.Frame,
        y: np.array,
        sample_weight: np.array = None,
        eval_set=None,
        sample_weight_eval_set=None,
        **kwargs,
    ):
        """Fit the model on training data and use optional validation data to tune parameters to avoid overfitting.
        Args:
            X (dt.Frame): training data, concatenated output of all active transformers' `fit_transform()` method
                Shape: (N, p), rows are observations, columns are features (attributes)
            y (np.array): training target values, numeric for regression, numeric or categorical for classification
                Shape: (N, ), 1 target value per observation
            sample_weight (np.array): (optional) training observation weight values, numeric
                Shape: (N, ), 1 observation weight value per observation
            eval_set (list(tuple(dt.Frame, np.array))): (optional) validation data and target values
                list must have length of 1, containing 1 tuple of X and y for validation data
                Shape: dt.Frame: (M, p), np.array: (M, )), same schema/format as training data, just different rows
            sample_weight_eval_set (list(np.array)): (optional) validation observation weight values, numeric
                list must have length of 1, containing 1 np.array for weights
                Shape: (M, ), 1 observation weight value per observation
            kwargs (dict): Additional internal arguments (see examples)
        Returns: None
        Note:
            Once the model is fitted, you can pass the state to Driverless AI via `set_model_properties()` for later
            retrieval during `predict()`. See examples.
            def set_model_properties(self, model=None, features=None, importances=None, iterations=None):
                :param model: model object that contains all large fitted objects related to model
                :param features: list of feature names fitted on
                :param importances: list of associated numerical importance of features
                :param iterations: number of iterations, used to predict on or re-use for fitting on full training data
        Recipe can raise h2oaicore.systemutils.IgnoreError to ignore error and avoid logging error for genetic algorithm.
        """

        self.config["time_column"] = self.params_base["time_column"]
        self.config["target_column"] = self.params_base["target"]
        self.config["tgc"] = self.params_base["tgc"]
        # self.config["unatt"] = self.params_base["ufapt"]
        self.config["unatt"] = []
        self.config["drop_columns"] = []
        self.config["forecast_horizon"] = self.params_base["pred_periods"]
        self.config["num_classes"] = self.params_base["num_classes"]

        if self.config["num_classes"] == 1:
            self.config["loss_function"] = torch.nn.MSELoss
        elif self.config["num_classes"] == 2:
            self.config["loss_function"] = torch.nn.BCELoss
        else:
            self.config["loss_function"] = torch.nn.CrossEntropyLoss

        if self.config["num_classes"] > 1:
            y = (
                pd.Series(y)
                .map({v: i for i, v in enumerate(self.params_base["labels"])})
                .values
            )

        train_df = X.to_pandas()
        train_df[self.config["time_column"]] = pd.to_datetime(
            train_df[self.config["time_column"]]
        )
        self.config["dilations"] = [
            z
            for z in self.config["dilations"]
            if 2 * z < train_df[self.params_base["time_column"]].nunique()
        ]
        self.config["min_seq_len"] = max(
            self.params_base["pred_periods"], np.max(self.config["dilations"])
        )
        train_df[self.params_base["target"]] = y

        dprep = DataPrep(self.config).fit(train_df)
        self.config = dprep.config

        train_ds = RNNDS_train(self.config, train_df, dprep)

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
            collate_fn=collator,
        )

        model = fit(TSRNN(self.config), train_loader, self.config)

        self.set_model_properties(
            model=model,
            features=list(X.names),
            importances=[1] * X.shape[1],
            iterations=-1,
        )
        self.train_df = train_df
        self.dprep = dprep

    def predict(self, X, **kwargs):
        """Make predictions on a test set.
        Use the fitted state stored in `self` to make per-row predictions. Predictions must be independent of order of
        test set rows, and should not depend on the presence of any other rows.
        Args:
            X (dt.Frame): test data, concatenated output of all active transformers' `transform()` method
                Shape: (K, p)
            kwargs (dict): Additional internal arguments (see examples)
        Returns: dt.Frame, np.ndarray or pd.DataFrame, containing predictions (target values or class probabilities)
            Shape: (K, c) where c = 1 for regression or binary classification, and c>=3 for multi-class problems.
        Note:
            Retrieve the fitted state via `get_model_properties()`, which returns the arguments that were passed after
            the call to `set_model_properties()` during `fit()`. See examples.
        Recipe can raise h2oaicore.systemutils.IgnoreError to ignore error and avoid logging error for genetic algorithm.
        """

        test_df = X.to_pandas()
        test_df[self.config["time_column"]] = pd.to_datetime(
            test_df[self.config["time_column"]]
        )
        test_ds = RNNDS_test(self.config, self.train_df, test_df, self.dprep)

        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
            collate_fn=collator,
        )

        model, _, _, _ = self.get_model_properties()
        preds = predict(model, test_loader, self.config)
        return preds
