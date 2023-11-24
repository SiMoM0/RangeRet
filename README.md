# RangeRet

3D Semantic Segmentation using Range Images and Retentive Network

## Training

Run the following commands, specifying the SemanticKITTI dataset path

```
cd RangeRet
python3 main.py <config_path> <data_path>
```

Example: `python3 main.py config/RangeRet-semantickitti.yaml /semanticKITTI/dataset/`

Checkpoint will be saved in `checkpoints/model-checkpoint.pt`, the final model will be saved as `rangeret-model.pt` and training logs in `log/` with the current date and time.

## Inference

Run the following script, specifying the SemanticKITTI dataset path

```
cd RangeRet
python3 infer.py <config_path> <model_path> <data_path> <pred_path> <split>
```

where split can be `train`, `valid` or `test`.

Example of inference on SemanticKITTI validation set: `python3 infer.py config/RangeRet-semantickitti.yaml rangeret-model.pt /semanticKITTI/dataset predictions/ valid`