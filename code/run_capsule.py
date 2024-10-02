""" top level run script """
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import pandas as pd

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.widgets as sw

from utils import compute_missing_metrics, apply_unit_classifier

# AIND
from aind_data_schema.core.processing import DataProcess

URL = "https://github.com/AllenNeuralDynamics/aind-ephys-unit-classifier"
VERSION = "1.0"

data_folder = Path("../data")
results_folder = Path("../results")

this_folder = Path(__file__).parent

n_jobs = -1
job_kwargs = dict(n_jobs=n_jobs)


if __name__ == "__main__":
    ####### UNIT CLASSIFIER ########
    print("UNIT CLASSIFIER")
    unit_classifier_params = {}
    unit_classifier_notes = ""
    t_unit_classifier_start_all = time.perf_counter()

    # find ecephys folder / or postprocessed
    data_process_prefix = "data_process_unit_classifier"

    si.set_global_job_kwargs(**job_kwargs)

    ecephys_sorted_folders = [
        p
        for p in data_folder.iterdir()
        if p.is_dir() and "ecephys" in p.name or "behavior" in p.name and "sorted" in p.name
    ]
    unit_classifier_model_folders = [
        p for p in this_folder.iterdir() if p.is_dir() and "unit_classifier_model" in p.name
    ]
    assert len(unit_classifier_model_folders) == 1
    unit_classifier_model_folder = unit_classifier_model_folders[0]
    unit_classifier_params["unit_classifier_model"] = unit_classifier_model_folder.name

    # load required metrics and model
    noise_neuron_pkl = unit_classifier_model_folder / "noise-neuron_classifier.pkl"
    sua_mua_pkl = unit_classifier_model_folder / "sua-mua_classifier.pkl"
    assert noise_neuron_pkl.is_file(), "Noise/Neuron model not found"
    assert sua_mua_pkl.is_file(), "SUA/MUA model not found"
    assert (unit_classifier_model_folder / "metrics.json").is_file(), "Required metrics not found"
    with open(unit_classifier_model_folder / "metrics.json", "r") as f:
        required_metrics = json.load(f)

    pipeline_mode = True
    if len(ecephys_sorted_folders) > 0:
        # capsule mode
        assert len(ecephys_sorted_folders) == 1, "Attach one sorted asset at a time"
        ecephys_sorted_folder = ecephys_sorted_folders[0]
        postprocessed_base_folder = ecephys_sorted_folder / "postprocessed"
        session_name = ecephys_sorted_folder.name[: ecephys_sorted_folder.name.find("_sorted")]
        pipeline_mode = False
    elif (data_folder / "postprocessing_pipeline_output_test").is_dir():
        print("\n*******************\n**** TEST MODE ****\n*******************\n")
        postprocessed_base_folder = data_folder / "postprocessing_pipeline_output_test"
    else:
        postprocessed_base_folder = data_folder

    if pipeline_mode:
        postprocessed_folders = [
            p for p in postprocessed_base_folder.iterdir() if "postprocessed_" in p.name and "postprocessed-sorting" not in p.name
        ]
    else:
        postprocessed_folders = [p for p in postprocessed_base_folder.iterdir() if p.is_dir()]

    for postprocessed_folder in postprocessed_folders:
        datetime_start_unit_classifier = datetime.now()
        t_unit_classifier_start = time.perf_counter()
        if pipeline_mode:
            recording_name = ("_").join(postprocessed_folder.stem.split("_")[1:])
        else:
            recording_name = postprocessed_folder.name
        if recording_name.endswith(".zarr"):
            recording_name = recording_name[:recording_name.find(".zarr")]
        unit_classifier_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"
        unit_classifier_output_csv_file = results_folder / f"unit_classifier_{recording_name}.csv"

        try:
            analyzer = si.load_sorting_analyzer_or_waveforms(postprocessed_folder)
            print(f"Applying unit classifier to recording: {recording_name}")
        except:
            print(f"Spike sorting failed on {recording_name}. Skipping unit classification")
            # create an mock result file (needed for pipeline)
            mock_df = pd.DataFrame()
            mock_df.to_csv(unit_classifier_output_csv_file)
            continue

        input_metrics = compute_missing_metrics(analyzer, required_metrics, n_jobs=n_jobs, verbose=True)

        prediction_df = apply_unit_classifier(
            metrics=input_metrics, noise_neuron_classifier_pkl=noise_neuron_pkl, sua_mua_classifier_pkl=sua_mua_pkl
        )

        decoder_label = prediction_df["decoder_label"]

        n_sua = int(np.sum(decoder_label == "sua"))
        n_mua = int(np.sum(decoder_label == "mua"))
        n_noise = int(np.sum(decoder_label == "noise"))
        n_units = int(len(analyzer.unit_ids))

        print(f"\tNOISE: {n_noise} / {n_units}")
        print(f"\tSUA: {n_sua} / {n_units}")
        print(f"\tMUA: {n_mua} / {n_units}")

        unit_classifier_notes += f"NOISE: {n_noise} / {n_units}\n"
        unit_classifier_notes += f"SUA: {n_sua} / {n_units}\n"
        unit_classifier_notes += f"MUA: {n_mua} / {n_units}\n"

        prediction_df.to_csv(unit_classifier_output_csv_file, index=False)

        t_unit_classifier_end = time.perf_counter()

        elapsed_time_unit_classifier = np.round(t_unit_classifier_end - t_unit_classifier_start, 2)

        # save params in output
        unit_classifier_params["recording_name"] = recording_name

        unit_classifier_outputs = dict(
            total_units=n_units, noise_units=n_noise, neuronal_units=n_units - n_noise, sua_units=n_sua, mua_units=n_mua
        )

        if pipeline_mode:
            unit_classifier_process = DataProcess(
                name="Ephys curation",
                software_version=VERSION,  # either release or git commit
                start_date_time=datetime_start_unit_classifier,
                end_date_time=datetime_start_unit_classifier
                + timedelta(seconds=np.floor(elapsed_time_unit_classifier)),
                input_location=str(data_folder),
                output_location=str(results_folder),
                code_url=URL,
                parameters=unit_classifier_params,
                outputs=unit_classifier_outputs,
                notes=unit_classifier_notes,
            )
            with open(unit_classifier_output_process_json, "w") as f:
                f.write(unit_classifier_process.model_dump_json(indent=3))

    t_unit_classifier_end_all = time.perf_counter()
    elapsed_time_unit_classifier_all = np.round(t_unit_classifier_end_all - t_unit_classifier_start_all, 2)
    print(f"UNIT CLASSIFIER time: {elapsed_time_unit_classifier_all}s")
