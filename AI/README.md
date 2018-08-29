# AI geoprocessing scripts
A collection of python (v3.7) scripts used to process SAR images.
* `map_envi.py` provides simple ENVI format loading and saving procedures.
* `map_adfilter.py` provides a procedure to fix "broken" pixels as well as an example of fix_pixels + anisotropic_diffusion filtering of a composite SAR image.
* `map_assemble_tensor.py` loads a number of ENVI images, performs some filtering and/or cropping and assembles a numpy tensor (ndarray) from them, according to instructions read from `recipe.json` file that should reside in `data` directory along with ENVI images.
* `map_process.py` performs further data filtering, clusterization with k-means, folowwed by fitting of gaussian mixture model on the given tensor (instructions for additional filtering are drown from same `recipe.json`). The trained model is than used to assign label probabilities to each image pixel.
* `map_vis.py` is a template for a data (clusterization result) post-processing, visualisation and saving procedure.


## License
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg?longCache=true&style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
