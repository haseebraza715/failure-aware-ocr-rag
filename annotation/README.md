# FAAR Annotation Study

This directory is populated only from a real B0 baseline output. Run the fixed plan commands:

```bash
python sample_failures.py --baseline results/b0.json --n 100 --out annotation/
python extract_ocr.py --engine got-ocr-2 --samples annotation/samples.json --out annotation/ocr_texts/
```

Create one Label Studio project using `annotation/label_studio/config.xml`, then import `annotation/label_studio/tasks.json`. Two people must independently annotate every task with exactly one of `semantic`, `word_level`, `structural`, or `other`. Export each person's completed annotations separately and run:

```bash
python compute_kappa.py --annotator1 annotation/annotator1.json --annotator2 annotation/annotator2.json --out annotation/kappa.json
```

The script exits unsuccessfully when Cohen's kappa is below `0.65`. It does not generate labels.
