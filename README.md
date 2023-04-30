# Automatic reformatting to LV short-axis view (SAX)
Author: Marta Nuñez-Garcia (marnugar@gmail.com)

## About
Implementation of the method described in:
[*Automatic multiplanar CT reformatting from trans-axial into left ventricle short-axis view*. Marta Nuñez-Garcia et al. STACOM (2020)](https://link.springer.com/chapter/10.1007/978-3-030-68107-4_2). Please cite this reference when using this code. PDF available here: [hal.inria.fr](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjM2O2xqaz1AhUP2BQKHR95AyIQFnoECAsQAQ&url=https%3A%2F%2Fhal.inria.fr%2Fhal-02961500%2Fdocument&usg=AOvVaw2t4ZjZm5ZgfdZa1cxhlp8w)

Given a raw trans-axial (TA) CT image and the corresponding LV endo, LV wall and RV epi segmentations (.mha or .vtk), compute LV short axis view image. It also reformats the masks.

Get meshes from masks and use them to find the transformation that aligns:
  1. MV plane
  2. Septum (RV position with regard to LV position)
  3. LV long axis
to the corresponding theoretical planes in standard short-axis view.
Manually set image parameters in the beginning: image size (nb of voxels), spacing, keep_physical_location = True/Fals

Resampling diagram:

![resampling diagram](https://github.com/martanunez/SAX_reformatting/blob/main/diagram_resampling.png)

Schematic pipeline:

![Schematic pipeline](https://github.com/martanunez/SAX_reformatting/blob/main/schematic_pipeline.png)

## Note
With respect to the method presented in the paper, this code additionally includes:
  - a 4th rotation (suggested and implemented by Nicolas Cedilnik) that improves LV septum alignment: after a  preliminary reformat to sax, use LV endo and LV epi masks (a slice midway along the long axis) to compute LV and RV centers and get the rotation matrix that will place the RV to the left of the LV.
  - The option of keeping the image world coordinates so they match in TA and SAX (default is TRUE).
  - The option of performing an initial automatic check of complete LV in TA view. If check_lv_cropped = True, exit if not complete (cropped) LV. 
  - Automatic check of potential appex cropping with current spacing and spacing modification if necessary.

## Extras
A couple of additional functionalities are also included:
- Basic Quality Control (QC) of the result: check final LV long axis direction (on a slightly different mesh) and compare it to the theoretical, expected, one.
- 17-AHA LV wall parcellation computation (compute_17_aha_segments_LVwall.py). The division is done according to the ["official" definition](https://www.pmod.com/files/download/v34/doc/pcardp/3615.htm), notably, taking into account that "only slices containing myocardium in all 360° are included", i.e. part of the base is excluded.  
- 17 regions LV mesh parcellation computation (compute_17_segments_mesh.py). Given a LV wall segmentation and corresponding mesh (endo, epi, midwall etc) compute 17-AHA segmentation projecting the division on the LV wall image to the mesh. Additionally compute alternative parcellation directly on the mesh (17 regions but not 17-AHA since this one fully includes the basal region).

## Important note on data display
A few reformatted images may show problems when displayed by some tools (i.e. ITK-SNAP error message: Failed to load image (...). Image has an invalid orientation (code XYZ)).
In those cases, images can be displayed using 3D Slicer, for instance, ticking the option "Ignore Orientation". Still the XYZ axes may be displayed in an odd manner (views can appear flipped). The orientation can be manually set to an easier-to-display one by using the code in workaround_visualization_issue.py but, importantly, the matching world coordinates with initial TA image will be lost (not relevant in most situations...).
Not sure if it is worthy to by default apply the workaround to all cases since this issue does not happen very often.



## Code
[Python](https://www.python.org/)

Required packages: VTK, SimpleITK, NumPy, pyvista, pyacvd. 

## Instructions
[Recommended] Create specific conda environment:
```
conda create --name sax_reformatting python=3.8
conda activate sax_reformatting
conda install -c simpleitk simpleitk
conda install -c anaconda numpy
pip install vtk    # conda not good behaviour
pip install pyvista
pip install pyacvd
```


Clone the repository:
```
git clone https://github.com/martanunez/SAX_reformatting

cd SAX_reformatting
```


Install the package sax_reformatting:
```
pip install -r requirements.txt
python3 -m pip install --upgrade build
python3 -m build
python -m pip install .
```

## Usage
```
python main.py  [-h] [--path PATH] [--ct_im FILENAME] 
                [--mask_lvendo FILENAME] [--mask_lvwall FILENAME] [--mask_rvepi FILENAME] 
                [--save SAVE] [--isotropic ISO]

Arguments:
  -h, --help          Show this help message and exit
  --path              Path to folder with input data
  --ct_im             Input CT image name
  --mask_lvendo       Input LV endo mask name
  --mask_lvwall       Input LV wall mask name
  --mask_rvepi        Input RV epi mask name
  --save              Save intermediate results (rotated meshes etc.)
  --isotropic         If output must be isotropic
```

## Usage example 1: reformat image to short-axis view (elapsed time: 17 s)
```
python main.py --path example_pat0/ --ct_im ct.mha --mask_lvendo ct-lvendo.mha --mask_lvwall ct-lvwall.mha --mask_rvepi ct-rvepi.mha
```

## Usage example 2: reformat image to short-axis view + compute corresponding 17-AHA segments + get epicardial mesh and parcellate it (17-AHA and 17 regions including full basal part)(elapsed time: 64 s)
```
python main.py --path example_pat0/ --ct_im ct.mha --mask_lvendo ct-lvendo.mha --mask_lvwall ct-lvwall.mha --mask_rvepi ct-rvepi.mha

python reformat_masks_to_SAX.py --path example_pat0/ --ct_im_sax ct-sax.mha --R_filename ct-R-matrix-sax.txt --mask_lvendo ct-lvendo.mha --mask_lvwall ct-lvwall.mha --mask_rvepi ct-rvepi.mha --mask_lvepi ct-lvepi-sax.mha

python compute_17_aha_segments_LVwall.py --path example_pat0/ --mask_lvendo_sax ct-lvendo-sax.mha --mask_lvwall_sax ct-lvwall-sax.mha --mask_rvepi_sax ct-rvepi-sax.mha --mask_lvepi_sax ct-lvepi-sax.mha

python compute_17_segments_mesh.py --path example_pat0/ --ct_mask ct-lvepi-sax.mha --ct_lvwall_labels_mask ct-lvwall-sax-dil-aha.mha

```

## License
The code in this repository is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details: [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/)
