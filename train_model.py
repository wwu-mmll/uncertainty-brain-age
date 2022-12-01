import os
import typer

from sklearn.metrics import median_absolute_error
from sklearn.model_selection import KFold

from photonai.base import PhotonRegistry, Hyperpipe, PipelineElement


def fetch_your_data():
    """
    This function is a placeholder for your data loading function.
    The model expects gray matter filtered data, therefore you should load data which is already gray matter masked
    or add a gray matter mask to the pipeline in main.
    """
    raise NotImplementedError('Please implement you own data-loading function')


def photon_main():
    # ###### register model
    base_folder = os.path.dirname(os.path.abspath(__file__))
    # custom_elements_folder = os.path.join(base_folder, 'wrapper_backup/')
    custom_elements_folder = os.path.join(base_folder, '')

    registry = PhotonRegistry(custom_elements_folder=custom_elements_folder)
    registry.delete('MccModel')
    registry = PhotonRegistry(custom_elements_folder=custom_elements_folder)
    registry.register(photon_name='MccModel',
                      class_str='MCCQRNN_Regressor.MCCQRNN_Regressor', element_type='Estimator')
    # check registration
    registry.info('MccModel')

    # ##### define pipeline
    my_pipe = Hyperpipe('MCC', KFold(n_splits=10),
                        metrics=['mean_absolute_error', median_absolute_error],
                        best_config_metric='mean_absolute_error',
                        cache_folder='./cache/',
                        eval_final_performance=False,
                        verbosity=2)

    my_pipe += PipelineElement('VarianceThreshold')
    my_pipe += PipelineElement('StandardScaler')
    my_pipe += PipelineElement('MccModel')

    # ##### load data and fit pipeline
    X, y = fetch_your_data()
    my_pipe.fit(X, y)


if __name__ == '__main__':
    typer.run(photon_main)
