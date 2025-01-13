# Fixing the Scale and Shift in Monocular Depth For Camera Pose Estimation

## Introduction
This repository hosts the solvers of the paper: <br />
"Fundamental Matrix Estimation Using Relative Depths" ECCV 2024. [[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08966.pdf)][[Supp](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08966-supp.pdf)] <br />
"Fixing the Scale and Shift in Monocular Depth For Camera Pose Estimation" Arxiv 2025. [[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08966.pdf)] <br />
We used [[graph-cut ransac](https://github.com/danini/graph-cut-ransac)] for robust estimation. <br />

## Installation
Install in Conda environment 
```bash
conda conda create -n posemono python=3.9
conda activate posemono
cd graph-cut-ransac
python -m pip install .
cd ..
pip install -r requirements.txt
```
Run the test code by specifically typing the scene, solver and depth name, _e.g._,
```bash
python test_calibrated.py --path 'pathtodata' --scene 'british_museum' --solver 'mono' --depth 'unidepth'
```
which solves the calibrated relative pose problem using 3 points with their monodepths.

## Data
An example data can be downloaded [[here](https://drive.google.com/file/d/13ZRI8D5gxLi37xbjH0lNJ3tk6QWr0MuA/view?usp=drive_link)]
More details come soon.

## Solvers in this repo
:file_folder: ``graph-cut-ransac/src/pygcransac/include/estimators``: contains the code of all the solvers. 
### Fundamental matrix estimation using relative depth with unknown relative scale:

solver_fundamental_matrix_4p4d - general fundamental matrix estimation
solver_fundamental_matrix_4p3d - varying focal lengths problem
solver_fundamental_matrix_3p3d - equal and unknown focal length


### Camera pose estimation using monocular depth modeling scale and shift:

Calibrated case:
solver_essential_matrix_mono_suv - closed-form solution, fastest. - 3PT$_{suv}$(C)
solver_essential_matrix_mono_suv_GB  - Gröbner basis solution. - 3PT$_{suv}$(GB)
solver_essential_matrix_mono_suv_Eigen - polynomial eigenvalue solution. - 3PT$_{suv}$(Eigen)
solver_essential_matrix_mono_suv_inverse - inverse depth model (not practical). - 3PT$_{suv}$(inverse) 

| Solver           | G-J | Eigen | Poly | Time($\mu$s) |
| :---------------: | :------: | :----: | :-------: | :-------: | 
| 3PT$_{suv}$(GB)       | $12\times 16$ |  $4\times 4$ | - | 4.45 | 
| 3PT$_{suv}$(Eigen)      | $6\times 10$ |  $4\times 4$ | - | 3.42 | 
| 3PT$_{suv}$(C)       | $3\times 6$ |  - | 4 | 1.46 | 
| 3PT$_{suv}$(inverse)        |  $54\times 66$ |  $12\times 12$ | - | 36.9 | 

Equal and unknown focal length:
solver_fundamental_matrix_mono_suvf_GB  - Gröbner basis solution. - 4PT$_{suv}f$(GB)
solver_fundamental_matrix_mono_suvf_Eigen - polynomial eigenvalue solution. - 4PT$_{suv}f$(Eigen)

| Solver           | G-J | Eigen | Poly | Time($\mu$s) |
| :---------------: | :------: | :----: | :-------: | :-------: | 
| 4PT$_{suv}f$(GB)       | $24\times 32$ |  $8\times 8$ | - | 12.5 | 
| 4PT$_{suv}f$(Eigen)      | $6\times 8$ |  $2\times 2$ | - | 2.38 | 

Varying focal lengths:
solver_fundamental_matrix_mono_suvfvar_GB  - Gröbner basis solution. - 4PT$_{suv}f_{1,2}$(GB)
solver_fundamental_matrix_mono_suvfvar_Eigen - polynomial eigenvalue solution. - 4PT$_{suv}f_{1,2}$(Eigen)

| Solver           | G-J | Eigen | Poly | Time($\mu$s) |
| :---------------: | :------: | :----: | :-------: | :-------: | 
| 4PT$_{suv}f_{1,2}$(GB)       | $20\times 24$ |  $4\times 4$ | - | 6.45 | 
| 4PT$_{suv}f_{1,2}$(Eigen)      | $6\times 8$ |  $2\times 2$ | - | 2.49 | 


## References
```BibTeX
@inproceedings{ding2025fundamental,
  title={Fundamental matrix estimation using relative depths},
  author={Ding, Yaqing and V{\'a}vra, V{\'a}clav and Bhayani, Snehal and Wu, Qianliang and Yang, Jian and Kukelova, Zuzana},
  booktitle={European Conference on Computer Vision},
  pages={142--159},
  year={2025},
  organization={Springer}
}
```
```BibTeX
@inproceedings{ding2025fixing,
  title={Fixing the Scale and Shift in Monocular Depth For Camera Pose Estimation},
  author={Ding, Yaqing and V{\'a}vra, V{\'a}clav and Kocur, Viktor and Yang, Jian and Sattler, Torsten and Kukelova, Zuzana},
  booktitle={arxiv},
  year={2025}
}
```