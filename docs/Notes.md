# Limitations

## Data

Currently using EDINA downloads, which provides binarized images (0 or 1); this reduces the overall fidelity of the image quality.

## Models

### SAM

Can't use individual points from GB1900 Gazetteer as text is often cut off due to small image sizes (288x288).

Filtered masks to those including points from GB1900 Gazetteer, this reduces the chance of SAM detecting useful text masks, however given that the masks were predominantly useless anyway the risk of this seems rather low.

Saved the outputs as pngs.