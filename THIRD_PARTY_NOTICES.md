# Third-Party Notices

This project depends on external open-source and model assets. Those assets are
not vendored in this repository unless explicitly noted.

## FunASR

The ASR, VAD, punctuation, and speaker workflows are built around FunASR runtime
dependencies and ModelScope-hosted models.

One utility module, `code/service/backend/utils/speaker_diarization_utils.py`,
contains adapted speaker diarization helper logic attributed in that file to
FunASR:

- Source: https://github.com/modelscope/FunASR
- Copyright: Alibaba, Inc. and its affiliates.

Use of upstream code and models is subject to their own licenses and model
terms. Review them before redistribution or commercial deployment.

## Model Artifacts

Model files are downloaded by `pretrained_models/download_models.py` into the
local `pretrained_models/` directory and are intentionally ignored by Git.
Users are responsible for complying with each model's license and terms of use.

