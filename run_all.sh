#!/usr/bin/env bash
# Reproduce every result in the repository (run from the project root).
set -euo pipefail
PY=${PY:-python}
$PY -m Reet.candidate_study   | tee Reet/results/candidate_study_log.txt
$PY -m Reet.build_dataset     | tee Reet/results/build_dataset_log.txt
$PY -m Reet.eda               | tee Reet/results/eda_log.txt
$PY -m Manasa.train_models    | tee Manasa/results/train_log.txt
$PY -m Manasa.tune            | tee Manasa/results/tune_log.txt
$PY -m Rekha.tune_decoder     | tee Rekha/results/tune_decoder_log.txt
$PY -m Rekha.evaluate_hmm     | tee Rekha/results/evaluate_log.txt
$PY -m Rekha.maps
$PY -m Sourish.report.make_report
