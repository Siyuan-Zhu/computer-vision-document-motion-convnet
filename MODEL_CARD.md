# Model card: document motion ConvNet

## Model

- **Task:** binary image classification. It labels a low-resolution image of a
  document `moving` (1) or `still` (0).
- **Architecture:** a ConvNet with four convolution layers (11 × 11, 5 × 5,
  1 × 1, 1 × 1), batch normalisation, max pooling, two dense layers of 1,024
  units with 20% dropout, and a sigmoid output. It has 11.9 million parameters.
- **Input:** a 100 × 70 × 3 image, preprocessed as described in
  [README: Preprocessing](README.md#preprocessing).
- **Output:** the probability that the page is moving. Above 0.5 counts as
  moving.
- **Implementations:** Keras 3 (runs on TensorFlow, JAX or PyTorch) and a
  native PyTorch version. Given the same weights, they agree to within
  0.00001.
- **Author and date:** Siyuan David Zhu, 2022. Packaged as a library in 2026.

## Intended use

- **Intended:** the capture trigger in a bulk document-scanning app. The app
  waits until the page is still, then takes the full-resolution shot.
- **Not intended:** any decision about people, or document content of any
  kind. It does not read text and has not been tested beyond the capture setup
  it was trained on.

## Data

- **Training:** 2,392 labelled images from a residency project's image set.
- **Test:** a separate set of 597 images: 290 moving and 307 still.
- **Not shipped:** the images belong to the programme and are not in this
  repo. No trained weights are shipped either.

## Results (2022)

- **5-fold cross-validation accuracy on the training set:** 96.2% (std 3.3).
  The folds scored 96.9%, 98.1%, 99.8%, 96.4% and 90.0%.
- **Held-out test set:** accuracy 0.98, with F1 0.98 for both classes.
  - Moving: precision 0.99, recall 0.98.
  - Still: precision 0.98, recall 0.99.

## Limitations and risks

- The cross-validation score may be optimistic, because images taken moments
  apart can end up in both training and validation folds. The
  separate test set is the more reliable figure.
- All images come from one data source and capture setup. Other devices,
  lighting conditions and paper types may lower accuracy.
- Each image is judged by itself. A misfire costs a blurred capture or a short
  delay.

## Reproducing

```bash
doc-motion train --data <folder with train/ and test/> --out runs/
```

Every run records its config, fold scores, training loss and test report in
`runs/<date>_<time>_<backend>/metrics.json`.
