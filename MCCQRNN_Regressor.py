import numpy as np
import keras.backend as K
from scipy import interpolate
from numpy.random import seed
from keras.models import Model
from scipy.stats import uniform
import tensorflow_probability as tfp
from keras.layers.core import Activation, Dense
from keras.layers import Input, BatchNormalization, Dropout

from photonai.modelwrapper.keras_base_models import KerasDnnBaseModel, KerasBaseRegressor


class MCCQRNN_Regressor(KerasDnnBaseModel, KerasBaseRegressor):
    def __init__(self, hidden=[32], epochs=10, optimizer='adam', learning_rate=.01, dropout=.2,
                 quantile_fits=None, y_transform=None):
        self.hidden = hidden[0]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.optimizer = optimizer
        self.nn_batch_size = 64
        self.model = None
        self.y_transform = y_transform
        self.nq = 101
        if quantile_fits is None:
            self.quantile_fits = np.asarray([q/self.nq for q in range(self.nq)])
        else:
            self.quantile_fits = np.asarray(quantile_fits)
        self.n_outputs = len(self.quantile_fits)
        self.set_random_seed = True     # control numpy.random during PREDICTION only
        print(self.n_outputs)

    def tilted_loss(self, y, f):
        e = (y - f)
        filter = K.cast(e > 0, 'float32')
        sec_filter = K.cast(e <= 0, 'float32')
        ls = K.mean(self.quantile_fits * filter * e +
                    (self.quantile_fits - np.ones(self.quantile_fits.shape)) * sec_filter * e)
        return tfp.stats.percentile(ls, 50.0, interpolation='midpoint')

    def get_model_dropout(self, X):
        inputs = Input(shape=X.shape[1:])
        x = Dense(self.hidden)(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x, training=True)
        output = Dense(self.n_outputs)(x)

        model = Model(inputs=inputs, outputs=output, name="DO_CQR_regressor")
        model.compile(loss=self.tilted_loss, optimizer=self.optimizer)
        return model

    def fit(self, X, y):
        print('fitting')
        self.model = self.get_model_dropout(X=X)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.nn_batch_size,
                       shuffle=True,
                       verbose=2)
        return self

    def predict(self, X, n_draws=1000):
        print('predicting')
        val_dict = {}
        quantiles = np.asarray([q / 100 for q in range(self.nq)])
        if self.set_random_seed:
            seed(42)    # set np random seed
        uni_rand = np.asarray([uniform.rvs() for _ in range(n_draws)])  # draws from uniform distribution
        for id in ['_noEpistemic', '_epistemic']:
            self.model._layers[4]._inbound_nodes[0].arguments['training'] = True
            q_preds_aleatory = list()
            q_preds_noAleatory = list()
            # Loop modeling epistemic uncertainty
            if id == '_epistemic':
                for i in range(n_draws):
                    y_pred = self.model.predict(X)
                    interp_cdf = interpolate.interp1d(self.quantile_fits,
                                                      # with dropout at test time
                                                      y_pred,
                                                      fill_value='extrapolate')

                    q_preds_aleatory.append(interp_cdf(uni_rand[i]))
                    q_preds_noAleatory.append(interp_cdf(.5))

                    if i % 100 == 0:
                        print('Drawing with Epistemic Uncertainty ' + str(i+1) + '/' + str(n_draws) + ' ' + id[1:])

                # build output
                val_dict = self.fill_val_dict(quantiles, val_dict, np.asarray(q_preds_aleatory), '_aleatory' + id)
                val_dict = self.fill_val_dict(quantiles, val_dict, np.asarray(q_preds_noAleatory), '_noAleatory' + id)

            elif id == '_noEpistemic':
                self.model._layers[4]._inbound_nodes[0].arguments['training'] = False
                y_pred = self.model.predict(X)
                interp_cdf = interpolate.interp1d(self.quantile_fits,
                                                  # with dropout at test time
                                                  y_pred,
                                                  fill_value='extrapolate')

                q_preds_aleatory.append(interp_cdf(uni_rand))
                q_preds_noAleatory.append(interp_cdf(.5))
                print('No Epistemic Uncertainty.')
                # build output
                val_dict = self.fill_val_dict(quantiles,
                                              val_dict,
                                              np.asarray(np.squeeze(q_preds_aleatory)).transpose(),
                                              '_aleatory' + id)
                val_dict = self.fill_val_dict(quantiles,
                                              val_dict,
                                              np.asarray(q_preds_noAleatory),
                                              '_noAleatory' + id,
                                              do_quants=False)

        y_pred = val_dict['median_noAleatory_epistemic']

        # for PHOTON summary output
        val_dict["y_pred"] = y_pred
        return np.array([tuple(val_dict[key][i] for key in val_dict.keys()) for i in range(len(y_pred))],
                        dtype=[(key, np.float64) for key in val_dict.keys()])

    def fill_val_dict(self, quantiles, val_dict, q_preds, id, do_quants=True):
        val_dict["median" + id] = np.median(q_preds, axis=0)
        val_dict["mean" + id] = np.mean(q_preds, axis=0)
        val_dict["std" + id] = np.std(q_preds, axis=0)
        val_dict["median_absolute_deviation" + id] = np.median(np.abs(q_preds - np.median(q_preds, axis=0)), axis=0)
        q_out = np.quantile(a=q_preds, q=quantiles, axis=0)
        # check for quantile cross-over
        if np.all(np.diff(q_out) < 0):
            print('Quantile cross-over!')

        # build output dict
        if do_quants:
            for i, q_ID in enumerate(quantiles):
                val_dict["%.3f" % q_ID + id] = q_out[i, :]
        return val_dict
