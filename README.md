## Commands

1. Splitting TIFF files into separate PNGs:
```cmd
python scripts/tiff_to_png.py data/tiffs outputs/pngs
```

2. Extracting Gb1900 gazetteer points contained within the TIFF examples:
```cmd
python scripts/get_gb1900_examples.py outputs/pngs/control-points.gpkg data/gb1900_gazetteer_201807.csv
```

3. Converting prediction outputs from ToponymExtractor to polygon masks:
```cmd
python scripts/convert_ToponymExtractor_outputs.py outputs/toponym-extractor/raw/ToponymOutputs.pkl outputs/pngs/control-points.gpkg outputs/toponym-extractor/extracted outputs/toponym-extractor/extracted/ToponymExtractor-inference-errors.csv
```

4. Processing polygon mask predictions to minimize overlapping predictions
```cmd
python scripts/postprocess_ToponymExtractor_outputs.py data/tiffs outputs/toponym-extractor/extracted outputs/pngs/control-points.gpkg outputs/toponym-extractor/postprocessed/Predictions outputs/toponym-extractor/postprocessed/Suppressed outputs/toponym-extractor/postprocessed/Ambiguous
```

5. Processing ambiguous predictions masks
```cmd
python scripts/process_ambiguous_ToponymExtractor_predictions.py outputs/toponym-extractor/postprocessed/Ambiguous outputs/toponym-extractor/ambiguous/ambiguous.pkl outputs/toponym-extractor/postprocessed/Ambiguous/control-points.gpkg outputs/toponym-extractor/ambiguous/Refined outputs/toponym-extractor/ambiguous/Refined/ToponymExtractor-inference-errors.csv
```

6. Combine prediction masks
```cmd
python scripts/combine_predictions.py outputs/toponym-extractor/revised outputs/toponym-extractor/postprocessed/Predictions outputs/toponym-extractor/ambiguous/Refined
```

7. (Step 4 retry) Processing polygon mask predictions to minimize overlapping predictions
```cmd
python scripts/postprocess_ToponymExtractor_outputs_v2.py data/tiffs outputs/toponym-extractor/extracted outputs/pngs/control-points.gpkg outputs/toponym-extractor/postprocessed-v2/Predictions outputs/toponym-extractor/postprocessed-v2/Suppressed outputs/toponym-extractor/postprocessed-v2/Ambiguous
```

8. (Step 5 retry) Processing ambiguous predictions masks
```cmd
python scripts/process_ambiguous_ToponymExtractor_predictions.py outputs/toponym-extractor/postprocessed-v2/Ambiguous outputs/toponym-extractor/ambiguous-v2/ambiguous-predictions-v2.pkl outputs/toponym-extractor/postprocessed-v2/Ambiguous/control-points.gpkg outputs/toponym-extractor/ambiguous-v2/Refined outputs/toponym-extractor/ambiguous-v2/Refined/ToponymExtractor-inference-errors.csv
```

9. (Step 6 retry) Combine prediction masks
```cmd
python scripts/combine_predictions.py outputs/toponym-extractor/revised outputs/toponym-extractor/postprocessed-v2/Predictions outputs/toponym-extractor/ambiguous-v2/Refined
```

10. Convert combined predictions and manually labelled data back to ICDAR 2025 format

Combined predictions:
```cmd
python scripts\convert_to_ICDAR2025.py outputs/toponym-extractor/revised-v2 outputs/manual-labelling/processed outputs/toponym-extractor/revised-icdar/with-ground-truths.json
```

Manually Labelled:
```cmd
python scripts\convert_to_ICDAR2025.py outputs/manual-labelling/refined outputs/manual-labelling/processed outputs/manual-labelling/refined-icdar/ground-truths.json --ground-truth
```

11. Evaluate predictions using ICDAR 2025 evaluation metrics:
```cmd
python map-text-evaluation/eval.py --gt outputs/manual-labelling/refined-icdar/ground-truths.json --pred outputs/toponym-extractor/revised-icdar/with-ground-truths.json --output outputs/metrics/baseline-icdar-metrics.json --task detrecedges
```

12. Evaluate predictions using custom metrics:
```cmd
python scripts\custom_metrics.py outputs/toponym-extractor/revised-v2 outputs/manual-labelling/refined outputs/metrics/custom-icdar-metrics.json --tiff-dir outputs/manual-labelling/refined outputs/manual-labelling/processed
```

13. Build train/eval dataset:
```
python scripts/construct_training_data.py outputs/manual-labelling/processed outputs/manual-labelling/refined outputs/modelling
```
