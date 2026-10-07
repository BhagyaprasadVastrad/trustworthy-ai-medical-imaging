# Runnable source reanalysis

The `src/` package contains the reproducible implementation for the corrected analysis while the executed notebooks remain the research record.

## Workflow

1. Train five baseline models with validation-loss early stopping and basic augmentation.
2. Run aligned RSNA inference for seeds 42, 43, 44, 45, 46.
3. Repeat external evaluation, temperature scaling, and confidence analysis for all five seeds.
4. Run Stage 5 with identifier-checked ensemble predictions.
5. Compare disagreement with one-minus-max-probability and predictive entropy.
6. Bootstrap AUROC and error-rate confidence intervals.

## Commands

```bash
python -m src.train_baseline --seed 42
python -m src.train_baseline --seed 43
python -m src.train_baseline --seed 44
python -m src.train_baseline --seed 45
python -m src.train_baseline --seed 46

python -m src.reanalysis
python -m src.stage5
```

RSNA data are external and are not included in the repository.