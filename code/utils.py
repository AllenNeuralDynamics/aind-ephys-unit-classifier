import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
from pathlib import Path

import spikeinterface as si
import spikeinterface.postprocessing as spost
from spikeinterface.postprocessing.template_metrics import get_multi_channel_template_metric_names
import spikeinterface.qualitymetrics as sqm


default_qm_params = {
    "presence_ratio": {"bin_duration_s": 60},
    "snr": {"peak_sign": "neg", "peak_mode": "extremum", "random_chunk_kwargs_dict": None},
    "isi_violation": {"isi_threshold_ms": 1.5, "min_isi_ms": 0},
    "rp_violation": {"refractory_period_ms": 1, "censored_period_ms": 0.0},
    "sliding_rp_violation": {
        "bin_size_ms": 0.25,
        "window_size_s": 1,
        "exclude_ref_period_below_ms": 0.5,
        "max_ref_period_ms": 10,
        "contamination_values": None,
    },
    "amplitude_cutoff": {
        "peak_sign": "neg",
        "num_histogram_bins": 100,
        "histogram_smoothing_value": 3,
        "amplitudes_bins_min_ratio": 5,
    },
    "amplitude_median": {"peak_sign": "neg"},
    "amplitude_cv": {
        "average_num_spikes_per_bin": 50,
        "percentiles": (5, 95),
        "min_num_bins": 10,
        "amplitude_extension": "spike_amplitudes",
    },
    "firing_range": {"bin_size_s": 5, "percentiles": (5, 95)},
    "synchrony": {"synchrony_sizes": (2, 4, 8)},
    "nearest_neighbor": {"max_spikes": 10000, "n_neighbors": 4},
    "nn_isolation": {"max_spikes": 10000, "min_spikes": 10, "n_neighbors": 4, "n_components": 10, "radius_um": 100},
    "nn_noise_overlap": {"max_spikes": 10000, "min_spikes": 10, "n_neighbors": 4, "n_components": 10, "radius_um": 100},
    "silhouette": {"method": ("simplified",)},
}

qm_metrics_map = dict(
    isi_violations_ratio="isi_violation",
    isi_violations_count="isi_violation",
    rp_contamination="rp_violation",
    rp_violations="rp_violation",
    amplitude_cv_median="amplitude_cv",
    amplitude_cv_range="amplitude_cv",
    drift_ptp="drift",
    drift_std="drift",
    drift_mad="drift",
    sync_spike_2="synchrony",
    sync_spike_4="synchrony",
    sync_spike_8="synchrony",
    nn_hit_rate="nearest_neighbor",
    nn_miss_rate="nearest_neighbor"
)


def compute_missing_metrics(we, required_metrics, qm_params=None, verbose=False, n_jobs=1):
    """
    TODO
    """
    qm = we.load_extension("quality_metrics").get_data()
    tm = we.load_extension("template_metrics").get_data()

    all_metrics = qm.merge(tm, left_index=True, right_index=True)

    if qm_params is None:
        qm_params = default_qm_params 

    if all([m in all_metrics.columns for m in required_metrics]):
        if verbose:
            print("All metrics available")
        return all_metrics
    else:
        required_tm = [m for m in required_metrics if m in spost.get_template_metric_names()]
        missing_tm_metrics = []
        for m in required_tm:
            if m not in all_metrics.columns:
                missing_tm_metrics.append(m)
        missing_tm_metrics = list(np.unique(missing_tm_metrics))
        include_multi_channel_metrics = any([m in get_multi_channel_template_metric_names() for m in missing_tm_metrics])
        if verbose:
            print(f"Computing missing template metrics: {missing_tm_metrics}")

        tm_new = spost.compute_template_metrics(we, metric_names=missing_tm_metrics,
                                                include_multi_channel_metrics=include_multi_channel_metrics)

        required_qm = [m for m in required_metrics if m not in spost.get_template_metric_names()]

        missing_qm_metrics = []
        for m in required_qm:
            if m not in all_metrics.columns:
                new_metric = qm_metrics_map[m] if m in qm_metrics_map else m
                missing_qm_metrics.append(new_metric)
        missing_qm_metrics = list(np.unique(missing_qm_metrics))

        if verbose:
            print(f"Computing missing quality metrics: {missing_qm_metrics}")
        qm_new = sqm.compute_quality_metrics(we, metric_names=missing_qm_metrics, n_jobs=n_jobs,
                                             qm_params=qm_params)

        all_metrics = all_metrics.merge(qm_new, left_index=True, right_index=True)
        all_metrics = all_metrics.merge(tm_new, left_index=True, right_index=True)

        # re-sort with correct order
        all_metrics = all_metrics[required_metrics]

        return all_metrics

# V2
def apply_classifiers_v2(metrics, noise_neuron_classifier_pkl, sua_mua_classifier_pkl):
    # Load pickles
    with open(noise_neuron_classifier_pkl, 'rb') as file:
        noise_decoder = pickle.load(file)

    with open(sua_mua_classifier_pkl, 'rb') as file:
        sua_decoder = pickle.load(file)

    # Prepare input data
    input_data = metrics.copy(deep=True)
    input_data = input_data.astype('float32')
    input_data[np.isinf(input_data)] = np.nan

    original_test_index = input_data.index.values

    # Apply noise classifier
    noise_predictions = noise_decoder.predict(input_data)
    noise_prob  = noise_decoder.predict_proba(input_data)[:, 1]

    input_data['decoder_label'] = noise_predictions
    input_data['decoder_label'] = input_data['decoder_label'].map({1: 'noise', 0: 'neural'})
    input_data['decoder_probability'] = noise_prob

    # Filter rows where the noise prediction was not 1 (not noise)
    input_data_sua = input_data[input_data['decoder_label'] != 'noise']
    input_data_sua = input_data_sua.drop(columns=['decoder_label', 'decoder_probability'], axis=1)

    # Apply SUA classifier
    if not input_data_sua.empty:
        sua_predictions = sua_decoder.predict(input_data_sua)
        sua_prob = sua_decoder.predict_proba(input_data_sua)[:, 1]
        # set the probability to 1 - p for MUA
        sua_prob[sua_predictions == 0] = 1 - sua_prob[sua_predictions == 0]

        input_data_sua['decoder_label'] = sua_predictions
        input_data_sua['decoder_label'] = input_data_sua['decoder_label'].map({1: 'sua', 0: 'mua'})
        input_data_sua['decoder_probability'] = sua_prob

        # Update the original DataFrame with SUA predictions
        input_data.loc[input_data_sua.index, 'decoder_label'] = input_data_sua['decoder_label']
        input_data.loc[input_data_sua.index, 'decoder_probability'] = input_data_sua['decoder_probability']

    # Save the result to a CSV file
    input_data = input_data.set_index(original_test_index)

    return input_data[["decoder_label", "decoder_probability"]]


### V1
def apply_classifiers_v1(
    metrics,
    noise_neuron_classifier_pkl,
    sua_mua_classifier_pkl,
    noise_neuron_preprocess_pkl=None,
    sua_mua_preprocess_pkl=None,
    preprocess=False
):
    """
    TODO:
    """
    X_test_noise = metrics
    X_test_noise = X_test_noise.copy(deep=True)
    
    X_test_noise = get_clean_missing_vals_free_X(X_test_noise)
    new_X_test_noise = drop_columns(X_test_noise) 
    imputed_noise = impute_dataframe(new_X_test_noise)

    if preprocess and noise_neuron_preprocess_pkl is not None and noise_neuron_preprocess_pkl.is_file():
        with open(noise_neuron_preprocess_pkl, 'rb') as file:
            noise_neuron_preprocess = pickle.load(file) 
        X_test_noise_preprocessed = noise_neuron_preprocess.fit_transform(imputed_noise)

    with open(noise_neuron_classifier_pkl, 'rb') as file:
        noise_neuron_classifier = pickle.load(file)  

    y_pred = noise_neuron_classifier.predict(X_test_noise_preprocessed) #for every row \
    y_prob  = noise_neuron_classifier.predict_proba(X_test_noise_preprocessed)

    original_test_index = X_test_noise.index.values
    X_test_noise = X_test_noise.reset_index()
    # X_test_noise = X_test_noise.rename(columns={'index': 'cluster_id'})

    X_test_noise['noise_preds'] = y_pred
    X_test_noise['noise_preds'] = X_test_noise['noise_preds'].map({1 : 'noise', 0 :'neural'})
    X_test_noise['noise_probs'] = y_prob[:,1]

    #==================================================================================================
    # SUA
    X_test_sua = X_test_noise.loc[X_test_noise['noise_preds']== 'neural']
    X_test_sua = X_test_sua.drop(columns = ['noise_preds','noise_probs'],axis=1)
    X_test_sua = X_test_sua.copy(deep = True)

    X_test_sua = get_clean_missing_vals_free_X(X_test_sua)
    new_X_test_sua = drop_columns(X_test_sua) 
    imputed_sua = impute_dataframe(new_X_test_sua)
    
    if preprocess and sua_mua_preprocess_pkl is not None and sua_mua_preprocess_pkl.is_file():
        with open(sua_mua_preprocess_pkl, 'rb') as file:
            sua_mua_preprocess = pickle.load(file) 
        X_test_sua_preprocessed = sua_mua_preprocess.fit_transform(imputed_sua)

    with open(sua_mua_classifier_pkl, 'rb') as file:
        sua_mua_classifier = pickle.load(file)

    # X_test_sua = X_test_sua.reset_index()
    # X_test_sua = X_test_sua.rename(columns={'index': 'cluster_id'})

    y_pred = sua_mua_classifier.predict(X_test_sua_preprocessed) #for every row \
    y_prob  = sua_mua_classifier.predict_proba(X_test_sua_preprocessed)
    
    X_test_sua['sua_preds'] = y_pred
    X_test_sua['sua_preds'] = X_test_sua['sua_preds'].map({1 : 'sua', 0 :'mua'})
    X_test_sua['sua_probs'] = y_prob[:,1]
    
    # X_test_sua = X_test_sua.set_index(original_test_index)
    # X_test_noise = X_test_noise.set_index(original_test_index)
    
    X_test_noise.loc[X_test_sua.index, 'noise_preds'] = X_test_sua['sua_preds']
    X_test_noise.loc[X_test_sua.index, 'noise_probs'] = X_test_sua['sua_probs']

    X_test_noise = X_test_noise.rename(columns={'noise_preds': 'decoder_label'})
    X_test_noise = X_test_noise.rename(columns={'noise_probs': 'decoder_probability'})

    X_test_noise = X_test_noise.set_index(original_test_index)

    return X_test_noise[["decoder_label", "decoder_probability"]]


def drop_columns(dataframe, columns_to_drop=None):
    """
    Drops the specified columns from a pandas DataFrame. If columns_to_drop is None,
    it drops the default columns ('epoch_name', 'Var1', and 'Unnamed: 0') if they exist.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame from which columns will be dropped.
    columns_to_drop (list or None): A list of column names to be dropped. If None, default
        columns will be dropped.

    Returns:
    pd.DataFrame: The DataFrame with specified columns removed.
    """
    if columns_to_drop is None:
        default_columns_to_drop = ['epoch_name', 'Var1', 'Unnamed: 0']
        columns_to_drop = [col for col in default_columns_to_drop if col in dataframe.columns]

    return dataframe.drop(columns=columns_to_drop, axis=1)


def impute_dataframe(df):
    """
    Impute missing values in a DataFrame using the median for columns with <= 80% missing values, 
    and 0 for columns with > 80% missing values.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with missing values imputed.
    """
    threshold = 0.8  # Set the missing value threshold to 80%
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    for col in df.columns:
        missing_percentage = df[col].isna().mean()
        if missing_percentage <= threshold:
            # Impute using the median for columns with <= 80% missing values
            df[col].fillna(df[col].median(), inplace=True)
        else:
            # Impute with 0 for columns with > 80% missing values
            df[col].fillna(0, inplace=True)

    return df


def get_clean_missing_vals_free_X(dataframe):
    #X, y = split_data_to_X_y(dataframe)
    X = dataframe
    if 'group' in X.columns:
        X.drop(['group'],axis = 1,inplace=True)
    if 'cluster_id' in X.columns:
        X.drop(['cluster_id'],axis = 1,inplace=True)
    if 'target' in X.columns:
        X.drop(['target'],axis = 1,inplace=True)
    if 'class' in X.columns:
        X.drop(['class'],axis = 1,inplace=True)
    
    #X_complete = remove_miss_vals(X)
    return X

