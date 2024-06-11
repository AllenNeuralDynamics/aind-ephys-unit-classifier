# Unit classifier for AIND ephys pipeline
## aind-ephys-unit-classifier


### Description

This capsule is designed to automatically label spike sorted units for the AIND pipeline.

It uses pre-trained models to classify units as:
- noise (non-neuronal)
- MUA (multi-unit activity)
- SUA (single-unit activity)

The model was developed by the [Musall's Group](https://brainstatelab.wordpress.com/) at the Forschungszentrum Jülich,
Germany.

The model files are in the `code/unit_classifier_models_v1.0` folder.

- `metrics.json`: list of required metrics for the models
- `noise-neuron_classifier.pkl`: the `scikit-learn` model for noise *VS* neuron classification
- `sua-mua_classifier.pkl`: the `scikit-learn` model for sua *VS* mua classification


### Inputs

The `data/` folder must include the output of the [aind-ephys-postprocessing](https://github.com/AllenNeuralDynamics/aind-ephys-postprocessing), including the `postprocessed_{recording_name}` folder.

### Parameters

The `code/run` script takes no arguments. 


### Output

The output of this capsule is the following:

- `results/unit_classifier_{recording_name}.csv` file, containing the `decoder_labels` and `decoder_probability` for each unit
- `results/data_process_unit_classifier_{recording_name}.json` file, a JSON file containing a `DataProcess` object from the [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) package.

