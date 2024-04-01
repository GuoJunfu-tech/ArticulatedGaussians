# Related works
* [deformable-GS](https://ingra14m.github.io/Deformable-Gaussians/) | [Paper](https://arxiv.org/abs/2309.13101)
* [PARIS](https://github.com/3dlg-hcvc/paris) | [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_PARIS_Part-level_Reconstruction_and_Motion_Analysis_for_Articulated_Objects_ICCV_2023_paper.pdf)
* [SuGaR](https://github.com/Anttwo/SuGaR) | [Paper](https://arxiv.org/abs/2311.12775)


## Dataset

### Build data
Tools: [https://github.com/GuoJunfu-tech/build_data](https://github.com/GuoJunfu-tech/build_data)
(Readme there has not be finished.)

Articulated datasets: [Raw Data](https://aspis.cmpt.sfu.ca/projects/paris/dataset.zip) | [Alternatives](https://1sfu-my.sharepoint.com/:u:/g/personal/jla861_sfu_ca/EeEggZVIENFJm6ZEORQ8QwIBhhlY9El1amq8A9zLl0WQJA?e=Tfs0N7)

We organize the datasets as follows:

```shell
├── data
    | laptop_2_40
    | fridge_2_40
    | ...
    | ---- Below are from Deform-GS ----
│   | D-NeRF 
│     ├── hook
│     ├── standup 
│     ├── ...
│   | NeRF-DS
│     ├── as
│     ├── basin
│     ├── ...
│   | HyperNeRF
│     ├── interp
│     ├── misc
│     ├── vrig
```


## Pipeline

TODO



## Run

### Environment

```shell
git clone https://github.com/GuoJunfu-tech/ArticulatedGaussians --recursive
cd ArticulatedGaussians

conda create -n deformable_gaussian_env python=3.7 # TODO rename
conda activate deformable_gaussian_env

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install dependencies
pip install -r requirements.txt
```



### Train
```
python train.py -s ./data/laptop_2_40 -m output/test_revolute --is_blender
```


~~You can also **train with the GUI:**~~ (Don't)

```shell
python train_gui.py -s path/to/your/dataset -m output/exp-name --eval --is_blender
```

- click `start` to start training, and click `stop` to stop training.
- The GUI viewer is still under development, many buttons do not have corresponding functions currently. We plan to :
  - [ ] reload checkpoints from the pre-trained model.
  - [ ] Complete the functions of the other vacant buttons in the GUI.



### Render & Evaluation

Not implemented yet

