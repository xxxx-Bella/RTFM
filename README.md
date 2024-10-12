# RTFM
This repo contains the Pytorch implementation of our paper:
> [**Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning**](https://arxiv.org/pdf/2101.10030.pdf)
>
> [Yu Tian](https://yutianyt.com/), [Guansong Pang](https://sites.google.com/site/gspangsite/home?authuser=0), Yuanhong Chen, Rajvinder Singh, Johan W. Verjans, [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/).

- **Accepted at ICCV 2021.**  

- **SOTA on 4 benchmarks.** Check out [**Papers With Code**](https://paperswithcode.com/paper/weakly-supervised-video-anomaly-detection) for [**Video Anomaly Detection**](https://paperswithcode.com/task/anomaly-detection-in-surveillance-videos). 


## Training

### Setup

**Please download the extracted I3d features for ShanghaiTech and UCF-Crime dataset from links below:**

> [**ShanghaiTech train i3d onedirve**](https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/EiLi_oBQnAFCq3UG184p_akBLDBVdCqRNCzSDhbqpjFQXw?e=hBAexc)
> 
> [**ShanghaiTech test i3d onedrive**](https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/EvUUrWqpWqVHrXBzxbzAdD8BlgH1SICKQbmdVu7K5nR9xA?e=oWTk8G)
> 
> [**ShanghaiTech features on Google dirve**](https://drive.google.com/file/d/1-w9xsx2FbwFf96A1y1GFcZ3odzdEBves/view?usp=sharing)
> 
> [**checkpoint for ShanghaiTech**](https://drive.google.com/file/d/1epISwbTZ_LXKfJzfYVIVwnxQ6q49lj5B/view?usp=sharing)

**Extracted I3d features for UCF-Crime dataset**

> [**UCF-Crime train i3d onedirve**](https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/ErCr6bjDzzZPstgposv1ttYBjv_ZBsAbNTbwyl3yX8QCHA?e=BzNuJ2)
> 
> [**UCF-Crime test i3d onedrive**](https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/EsmBEpklrShEjTFOWTd5FooBkJR3DPxp3cIZN-R8b2hhLA?e=hlcZFO)
> 
> [**UCF-Crime train I3d features on Google drive**](https://drive.google.com/file/d/16LumirTnWOOu8_Uh7fcC7RWpSBFobDUA/view?usp=sharing)
> 
> [**UCF-Crime test I3d features on Google drive**](https://drive.google.com/drive/folders/1QCBTDUMBXYU9PonPh1TWnRtpTKOX-fxr?usp=sharing)
> 
> [**checkpoint for UCF-Crime**](https://uao365-my.sharepoint.com/:u:/g/personal/a1697106_adelaide_edu_au/Ed0gS0RZ5hFMqVa8LxcO3sYBqFEmzMU5IsvvLWxioTatKw?e=qHEl5Z)

The above features use the resnet50 I3D to extract from this [**repo**](https://github.com/Tushar-N/pytorch-resnet3d).

Follow previous works, we also apply 10-crop augmentations. 

The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the download datasets above in `list/shanghai-i3d-test-10crop.list` and `list/shanghai-i3d-train-10crop.list`.
- Feel free to change the hyperparameters in `option.py`
### Train and test the RTFM
After the setup, simply run the following commands: 

```shell
cd /home/featurize/work/yuxin/WVAD/RTFM && conda create -n "rtfm" python=3.9 && conda activate rtfm && pip install torch matplotlib scikit-learn tqdm wandb seaborn

cd /home/featurize/work/yuxin/WVAD/RTFM && conda activate rtfm
# conda info --envs

python main_0.py --run-name rtfm --max-epoch 5000 --batch-size 4 --scene all

python main.py --run-name x-wo-aggregator --max-epoch 15000 --batch-size 4 --scene all
python main.py --run-name x-wo-dual --max-epoch 15000 --batch-size 4 --scene all

# loss terms
python main.py --run-name x-l_cls --max-epoch 5000 --batch-size 4 --scene all  --lambda2 0
python main.py --run-name x-l_mean --max-epoch 5000 --batch-size 4 --scene all --lambda1 0 --beta 0
python main.py --run-name x-l_var --max-epoch 5000 --batch-size 4 --scene all --lambda1 0 --alpha 0
python main.py --run-name x-l_mean_var --max-epoch 5000 --batch-size 4 --scene all --lambda1 0
python main.py --run-name x-l_cls_mean --max-epoch 5000 --batch-size 4 --scene all --beta 0
python main.py --run-name x-l_cls_var --max-epoch 5000 --batch-size 4 --scene all  --alpha 0
python main.py --run-name x-l_no_ss --max-epoch 5000 --batch-size 4 --scene all  --lambda3 0 --lambda4 0

python main.py --run-name x-scene-all-2 --max-epoch 15000 --batch-size 4 --scene all

python main.py --run-name x-scene-bike-4 --max-epoch 7000 --batch-size 4 --scene Bike_Roundabout
python main.py --run-name x-scene-cross-3 --max-epoch 5000 --batch-size 1 --scene Crossroads
python main.py --run-name x-scene-farm-3 --max-epoch 3000 --batch-size 1 --scene Farmland_Inspection
python main.py --run-name x-scene-highway-3 --max-epoch 3000 --batch-size 1 --scene Highway
python main.py --run-name x-scene-railway-3 --max-epoch 2000 --batch-size 1 --scene Railway_Inspection
python main.py --run-name x-scene-solar-3 --max-epoch 2000 --batch-size 1 --scene Solar_Panel_Inspection
python main.py --run-name x-scene-vehicle-3 --max-epoch 3000 --batch-size 1 --scene Vehicle_Roundabout

# ['all', 'Bike_Roundabout', 'Crossroads', 'Farmland_Inspection', 'Highway', 'Railway_Inspection', 'Solar_Panel_Inspection', 'Vehicle_Roundabout']

93d9cb47ca8e71ecdf675438033ea06ebc9cfd9c
```
# Git Commit/Push
    git config --global user.email "hyx18390659623@163.com" && git config --global user.name "xxxx-Bella"

    # sudo chmod 777 /home/featurize/work/MyPaper/
    git commit -m ""
    git push origin main

    git reset --soft HEAD^  # cancel last commit
    git reset --soft HEAD~2  # cancel last 2 commits
    git reset  # cancel add
    

## Next
* python list/make_gt_da.py (done)

* verify_frame_equal_label (done)  # ./data/split_video.py

* drone_anomaly_new (done)  # split each video (window-size=600)

* generate i3d of drone_anomaly_new (done)

* npy in abnormal -->(copy) output/drone_anomaly_new  (done)

---

# rm 
    cd ./log
    find . -type d -name "*-2" -exec rm {}/ckpt-best.pt \;
    find . -type d -name "*-2" -exec rm {}/*.pickle \;

    find . -type d -name "*-1" -exec find {} -type f -name "*.pickle" \; > files_to_delete.txt && cat files_to_delete.txt | xargs rm
    
    # current path, file size, memory
    du -sh *   
    du -ah --max-depth=1  # including hidden files

    # Find all *.pickle files
    find . -type f -name "threshold*" > files_to_delete.txt   

    # Find all *.pickle files in dirs except "all-data"
    find . -type d -name "all-data" -prune -o -type f -name "*.pickle" -print > files_to_delete.txt  

    # Then remove
    cat files_to_delete.txt | xargs rm