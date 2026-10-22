# Ball-Yolo-A-Sport-Ball-Dataset
Our data is divided into three categories: football, tennis, and table tennis, each stored in separate archives.  
The dataset is constructed based on sports videos, all of which have been automatically desensitized using the YOLO26 model. 

All annotation files are stored at https://doi.org/10.5061/dryad.3bk3j9m13 on dryad.

In this dataset, the files in the Videos directory under each of the tableTennis, tennis, and footBall directories have a one-to-one correspondence with the files in the Labels directory. Please refer to the README file in each Video directory or at https://doi.org/10.5061/dryad.3bk3j9m13 for details.


In addition, considering that blurring may interfere with detection and recognition, we have informally released the original videos on GitHub. The original videos can be viewed from our unofficially released version at https://github.com/Zoutianjian/Ball-Yolo-A-Sport-Ball-Dataset.
The corresponding naming conventions for the videos are as follows:  

```
{
        "MergeDetectBlue1_0": "BMN_1",
        "MergeBaiduTrainNew_1": "BMN_0"
        'FanZhenDong_Ana2':"default_06",
        'WangYiXuan_Valid':"default_07", 
        'XinXiang_7-7':"default_08", 
        'XuXin_FE':"default_05", 
        'YiTeng_025_01_h264_15':"default_00", 
        'YiTeng_025_02_h264_15':"default_01", 
        'YiTeng_025_03_h264_15':"default_02", 
        'YiTeng_025_04_h264_15':"default_03", 
        'YiTeng_075_01_h264_15':"default_04",
        'Amateur_5_5_5_0':"tennis_0", 
        'America_Best_10_1':"tennis_1", 
        'America_Best_Move_1':"tennis_2", 
        'AriyaHim_MostStd_forHand_1':"tennis_3", 
        'AriyaHim_MostStd_forHand_2':"tennis_4",
        'AriyaHim_MostStd_forHand_3':"tennis_5", 
        'AriyaHim_MostStd_forHand_4':"tennis_6", 
        'Federer_Top_40_1':"tennis_7", 
        'Federer_Top_40_2':"tennis_8", 
        'Federer_Top_40_3':"tennis_9", 
        'France_Review_Red_Court_1':"tennis_10", 
        'Real_Tennis':"tennis_11", 
        'Serve_190_And_14-year-old-Girl':"tennis_12", 
        'ShunYi_Amaetur_4_0_0':"tennis_13", 
        'ShunYi_Amaetur_4_0_3':"tennis_14", 
        'Std_6_0_Confort_PullBall':"tennis_15",
        'Game_Juven_Udinese_Boardcast_1': "Game_1_1", 
        'Game_Juven_Udinese_Boardcast_2': "Game_1_2",
        'Game_Juven_Udinese_Boardcast_5': "Game_1_3", 
        'Game_Livepool_Wolf_Boardcast_1': "Game_2_1", 
        'Game_Livepool_Wolf_Boardcast_2': "Game_2_2", 
        'Game_Livepool_Wolf_Boardcast_6': "Game_2_3", 
        'Game_Livepool_Wolf_Boardcast_8': "Game_2_4", 
        'Game_Real_Mardrid_ManCity_Boardcast_1': "Game_3_1",
        'Game_Real_Mardrid_ManCity_Boardcast_2': "Game_3_2", 
        'Game_Real_Mardrid_ManCity_Boardcast_4': "Game_3_3", 
        'Game_Torino_ACMilan_Boardcast_1': "Game_4_1",
        'Game_Torino_ACMilan_Boardcast_2': "Game_4_2",
    }
```
Reference of YOLO26 (Bibtex)

```
@software{yolo26_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO26},
  version = {26.0.0},
  year = {2026},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```

