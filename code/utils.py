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
    nn_miss_rate="nearest_neighbor",
)


def compute_missing_metrics(analyzer, required_metrics, qm_params=None, verbose=False, n_jobs=1):
    """
    TODO
    """
    qm = analyzer.get_extension("quality_metrics").get_data()
    tm = analyzer.get_extension("template_metrics").get_data()

    all_metrics = qm.merge(tm, left_index=True, right_index=True)

    if qm_params is None:
        qm_params = default_qm_params

    if all([m in all_metrics.columns for m in required_metrics]):
        if verbose:
            print("All metrics available")
        # re-sort with correct order
        all_metrics = all_metrics[required_metrics]

        return all_metrics
    else:
        required_tm = [m for m in required_metrics if m in spost.get_template_metric_names()]
        missing_tm_metrics = []
        for m in required_tm:
            if m not in all_metrics.columns:
                missing_tm_metrics.append(m)
        missing_tm_metrics = list(np.unique(missing_tm_metrics))
        include_multi_channel_metrics = any(
            [m in get_multi_channel_template_metric_names() for m in missing_tm_metrics]
        )
        if verbose:
            print(f"Computing missing template metrics: {missing_tm_metrics}")

        tm_new = spost.compute_template_metrics(
            analyzer, metric_names=missing_tm_metrics, include_multi_channel_metrics=include_multi_channel_metrics
        )

        required_qm = [m for m in required_metrics if m not in spost.get_template_metric_names()]

        missing_qm_metrics = []
        for m in required_qm:
            if m not in all_metrics.columns:
                new_metric = qm_metrics_map[m] if m in qm_metrics_map else m
                missing_qm_metrics.append(new_metric)
        missing_qm_metrics = list(np.unique(missing_qm_metrics))

        if verbose:
            print(f"Computing missing quality metrics: {missing_qm_metrics}")
        qm_new = sqm.compute_quality_metrics(analyzer, metric_names=missing_qm_metrics, n_jobs=n_jobs, qm_params=qm_params)

        all_metrics = all_metrics.merge(qm_new, left_index=True, right_index=True)
        all_metrics = all_metrics.merge(tm_new, left_index=True, right_index=True)

        # re-sort with correct order
        all_metrics = all_metrics[required_metrics]

        return all_metrics


def apply_unit_classifier(metrics, noise_neuron_classifier_pkl, sua_mua_classifier_pkl):
    # Load pickles
    with open(noise_neuron_classifier_pkl, "rb") as file:
        noise_decoder = pickle.load(file)

    with open(sua_mua_classifier_pkl, "rb") as file:
        sua_decoder = pickle.load(file)

    # Prepare input data
    input_data = metrics.copy(deep=True)
    input_data = input_data.astype("float32")
    input_data[np.isinf(input_data)] = np.nan

    original_test_index = input_data.index.values

    # Apply noise classifier
    noise_predictions = noise_decoder.predict(input_data)
    noise_prob = noise_decoder.predict_proba(input_data)[:, 1]

    input_data["decoder_label"] = noise_predictions
    input_data["decoder_label"] = input_data["decoder_label"].map({1: "noise", 0: "neural"})
    input_data["decoder_probability"] = noise_prob

    # Filter rows where the noise prediction was not 1 (not noise)
    input_data_sua = input_data[input_data["decoder_label"] != "noise"]
    input_data_sua = input_data_sua.drop(columns=["decoder_label", "decoder_probability"], axis=1)

    # Apply SUA classifier
    if not input_data_sua.empty:
        sua_predictions = sua_decoder.predict(input_data_sua)
        sua_prob = sua_decoder.predict_proba(input_data_sua)[:, 1]
        # set the probability to 1 - p for MUA
        sua_prob[sua_predictions == 0] = 1 - sua_prob[sua_predictions == 0]

        input_data_sua["decoder_label"] = sua_predictions
        input_data_sua["decoder_label"] = input_data_sua["decoder_label"].map({1: "sua", 0: "mua"})
        input_data_sua["decoder_probability"] = sua_prob

        # Update the original DataFrame with SUA predictions
        input_data.loc[input_data_sua.index, "decoder_label"] = input_data_sua["decoder_label"]
        input_data.loc[input_data_sua.index, "decoder_probability"] = input_data_sua["decoder_probability"]

    # Save the result to a CSV file
    input_data = input_data.set_index(original_test_index)

    return input_data[["decoder_label", "decoder_probability"]]
