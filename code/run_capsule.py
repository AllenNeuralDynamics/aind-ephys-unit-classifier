""" top level run script """
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
import pickle
import time
from datetime import datetime
import json

import spikeinterface as si
import spikeinterface.widgets as sw

from aind_data_schema.core.processing import DataProcess

from utils import compute_missing_metrics, apply_classifiers_v1, apply_classifiers_v2


data_folder = Path("../data")
results_folder = Path("../results")
n_jobs = -1

job_kwargs = dict(n_jobs=n_jobs)

unit_classifier_params = {}
GENERATE_VISUALIZATION_LINK = True
LABEL_CHOICES = ["noise", "SUA", "MUA", "pSUA", "pMUA"]


if __name__ == "__main__":
     ####### UNIT CLASSIFIER ########
    print("UNIT CLASSIFIER")
    unit_classifier_notes = ""
    t_unit_classifier_start_all = time.perf_counter()

    # find ecephys folder / or postprocessed
    data_process_prefix = "data_process_classifier"
    
    si.set_global_job_kwargs(**job_kwargs)

    ecephys_sorted_folders = [p for p in data_folder.iterdir() if p.is_dir() and "ecephys" in p.name and "sorted" in p.name]
    unit_classifier_model_folders = [p for p in data_folder.iterdir() if p.is_dir() and "unit_classifier_model" in p.name]
    assert len(unit_classifier_model_folders) == 1
    unit_classifier_model_folder = unit_classifier_model_folders[0]
    unit_classifier_params["unit_classifier_model"] = unit_classifier_model_folder.name

    # load required metrics and model
    noise_neuron_pkl = unit_classifier_model_folder / "noise-neuron_classifier.pkl"
    sua_mua_pkl = unit_classifier_model_folder / "sua-mua_classifier.pkl"
    if (unit_classifier_model_folder / "metrics.json").is_file():
        print("Noise models V2")
        version = 2
        with open(unit_classifier_model_folder / "metrics.json", "r") as f:
            required_metrics = json.load(f)
    else:
        print("Noise models V1")
        version = 1
        with open(unit_classifier_model_folder / "noise-neuron_metrics.json", "r") as f:
            noise_neuron_metrics = json.load(f)

        if (unit_classifier_model_folder / "noise-neuron_preprocess.pkl").is_file():
            noise_neuron_preprocess_pkl = unit_classifier_model_folder / "noise-neuron_preprocess.pkl"
        else:
            noise_neuron_preprocess_pkl = None
        if (unit_classifier_model_folder / "sua-mua_preprocess.pkl").is_file():
            sua_mua_preprocess_pkl = unit_classifier_model_folder / "sua-mua_preprocess.pkl"
        else:
            sua_mua_preprocess_pkl = None
        required_metrics = list(set(noise_neuron_metrics).union(set(sua_mua_metrics)))

    capsule_mode = False
    if len(ecephys_sorted_folders) > 0:
        # capsule mode
        assert len(ecephys_sorted_folders) == 1, "Attach one sorted asset at a time"
        ecephys_sorted_folder = ecephys_sorted_folders[0]
        postprocessed_folder = ecephys_sorted_folder / "postprocessed"
        session_name = ecephys_sorted_folder.name[:ecephys_sorted_folder.name.find("_sorted")]
        capsule_mode = True
    elif (data_folder / "postprocessing_pipeline_output_test").is_dir():
        print("\n*******************\n**** TEST MODE ****\n*******************\n")
        postprocessed_folder = data_folder / "postprocessing_pipeline_output_test"
    else:
        postprocessed_folder = data_folder

    if capsule_mode:
        postprocessed_folders = [p for p in postprocessed_folder.iterdir() if p.is_dir()]
    else:
        postprocessed_folders = [p for p in postprocessed_folder.iterdir() if "postprocessed_" in p.name and "-sorting" not in p.name]

    for postprocessed_folder in postprocessed_folders:
        datetime_start_unit_classifier = datetime.now()
        t_unit_classifier_start = time.perf_counter()
        if capsule_mode:
            recording_name = postprocessed_folder.name
        else:
            recording_name = ("_").join(postprocessed_folder.name.split("_")[1:])
        unit_classifier_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"
        unit_classifier_output_csv_file = results_folder / f"noiseclassification_{recording_name}.csv"
        unit_classifier_viz_output_file = results_folder / f"viz_noiseclassification_{recording_name}.txt"

        print(f"Applying noise classification to recording: {recording_name}")

        we = si.load_waveforms(postprocessed_folder, with_recording=False)

        input_metrics = compute_missing_metrics(we, required_metrics, n_jobs=n_jobs, verbose=True)

        if version == 1:
            prediction_df = apply_classifiers_v1(
                metrics=input_metrics,
                noise_neuron_classifier_pkl=noise_neuron_pkl, 
                sua_mua_classifier_pkl=sua_mua_pkl,
                noise_neuron_preprocess_pkl= noise_neuron_preprocess_pkl,
                sua_mua_preprocess_pkl=sua_mua_preprocess_pkl,
                preprocess=True
            )
        else:
            prediction_df = apply_classifiers_v2(
                metrics=input_metrics,
                noise_neuron_classifier_pkl=noise_neuron_pkl, 
                sua_mua_classifier_pkl=sua_mua_pkl
            )

        decoder_label = prediction_df["decoder_label"]
        we.sorting.set_property("decoder_label", prediction_df["decoder_label"])
        we.sorting.set_property("decoder_probability", np.round(prediction_df["decoder_probability"], 3))

        n_sua = np.sum(decoder_label == 'sua')
        n_mua = np.sum(decoder_label == 'mua')
        n_noise = np.sum(decoder_label == 'noise')
        n_units = len(we.unit_ids)

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
            total_units=n_units,
            noise_units=n_noise,
            neuronal_units=n_units - n_noise,
            sua_units=n_sua,
            mua_units=n_mua
        )
        
        if not capsule_mode:
            unit_classifier_process = DataProcess(
                    name="Ephys curation",
                    software_version=VERSION, # either release or git commit
                    start_date_time=datetime_start_unit_classifier,
                    end_date_time=datetime_start_unit_classifier + timedelta(seconds=np.floor(elapsed_time_unit_classifier)),
                    input_location=str(data_folder),
                    output_location=str(results_folder),
                    code_url=URL,
                    parameters=unit_classifier_params,
                    outputs=unit_classifier_outputs,
                    notes=unit_classifier_notes
                )
            with open(curation_output_process_json, "w") as f:
                f.write(curation_process.json(indent=3))
        else:
            # capsule mode
            if GENERATE_VISUALIZATION_LINK:
                we.sorting.set_property("decoder_label", prediction_df["decoder_label"])
                we.sorting.set_property("decoder_prob", np.round(prediction_df["decoder_probability"], 3))
                unit_table_properties = ['decoder_label', 'decoder_prob']

                # TODO: grab from processing.json
                sorter_name = "kilosort2_5"

                try:
                    w = sw.plot_sorting_summary(
                        we,
                        max_amplitudes_per_unit=500,
                        unit_table_properties=unit_table_properties,
                        curation=True,
                        label_choices=label_choices,
                        figlabel=f"{session_name} - {recording_name} - {sorter_name} - Noise decoder summary",
                        backend="sortingview"
                    )
                    visualization_txt = w.url.replace('\\"', "%22")
                    visualization_txt = visualization_txt.replace('#', "%23")
                    # remove escape characters
                    unit_classifier_viz_output_file.write_text(visualization_txt)
                except Exception as e:
                    print("KCL error", e)

    t_unit_classifier_end_all = time.perf_counter()
    elapsed_time_unit_classifier_all = np.round(t_unit_classifier_end_all - t_unit_classifier_start_all, 2)
    print(f"NOISE CLASSIFICATION time: {elapsed_time_unit_classifier_all}s")
