# Limitations

## Data

Currently using EDINA downloads, which provides binarized images (0 or 1); this reduces the overall fidelity of the image quality.

## Models

### SAM

Can't use individual points from GB1900 Gazetteer as text is often cut off due to small image sizes (288x288).

Filtered masks to those including points from GB1900 Gazetteer, this reduces the chance of SAM detecting useful text masks, however given that the masks were predominantly useless anyway the risk of this seems rather low.

Saved the outputs as pngs.

### DNTextSpotter

Unable to get DNTextSpotter to work due to installation difficulties. The model required specific versions of pytorch, which seemed to be in conflict with the CUDA versions available through UoE undergraduate compute server. DNTextSpotter has been dropped from development due to time constraints.

### ToponymExtractor

Outputs from the ToponymExtractor are promising, however there are challenges around resolving the truncated text from overlapping segments between pngs. Polygons for truncated text in the overlapping image segments can be easily resolved, however the text labels are more challenging. ToponymExtractor does not specify the locations for each letter in a captured word, so determining which range of letters need to be discarded from which label is difficult. This was an unforseen issue that stalls development along the critical path.

There are 3 instances to consider with overlapping image segments:

1. Text is entirely contained within overlapping area, there are two predictions for the text instances, each coming from one of the pngs the text belongs to. To resolve multiple predictions on the same text instance, a high IOU threshold can be used to determine they are predictions for the same text instance, then if the predicted text labels are different, confidence score can be used to determine which label to preference.

2. Text instance is truncated in one png, but completely captured in the other instance; for this scenario the truncated segment will be discarded.

3. Text extends beyond the overlap section in both pngs, this will make the text instance truncated in both pngs and is where the problem of resolving text labels described above arises. When both predicted text instances are truncated, the area surrounding the merged mask of both predictions will be saved to it's own png and passed to the model again to created a new text prediction.

Need to determine which of the 3 categories above each prediction falls into.

Heuristics can be made for large text with long spaces between letters.

### Out-of-scope

- Exploratory work suggests that snippets around the boundaries of ambiguous pngs are predicted worse than copies of the snippets located more centrally within the PNG. This suggests positional bias coming from pad values used in convolutional layers. One way around this would be to build a smaller image and then pad along all sides using a "wrap" method to provide a dummy area that can still be utilized during convolution.
- Pack ambiguous images to new PNGs more efficiently using box-packing algorithms.
- Need to devise a method from grouping text that is truncated by tiff boundaries.


## Post manual labelling analysis

Why did I do so many sample images when I can't manually assess all of them?

## Finetuning

Is it really a good idea to use augmentation techniques? Given that we have a very
small training set to begin with and we are constraining the model to a particular
set of maps.

Resizing an image with such poor quality to begin with will further degrade features.

# Final report notes

Emphasise the project acts as a discovery so the code is built with a mind towards future use on a larger dataset.

Focus on operational use case by emphasizing scalability and merging predictions.

Critique the choice of DETRs, based on the fact that this impedes the solutions ability to properly leverage contextual information through word linkage

Welsh names actually contain letters that fall outside the standard vocabulary

Models were treated as black boxes, due to the risk of failure if trying to tinker with internal components, which would then yield no results to report on.

Reconstruction of text is susceptible to imperfect detection between different images.

Predicting on image patches and using augmentation techniques like random cropping may be interfering training, this may in part explain slow convergence.

The need for overlapping image snippets could be overcome entirely by predicting whether or not the word is truncated as an additional task for the TextSpotter, this could also speed up training.

In hindsight, could have included ICDAR 2024's metric for word recognition.

It would appear the cropping problem is a currently undocumented problem, possibly due to the relative youth of multi-task text spotting tools and the academic focus on performance against evaluation datasets over application

Should have tested new metrics on ICDAR evaluation dataset

# Word level experiment

Manually split pngs: 750x750 with 250 overlaps

DeepSolo (Through ToponymExtractor - inference.py): min_patch_resolution set to 750
