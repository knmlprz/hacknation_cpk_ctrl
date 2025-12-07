# hacknation_cpk_ctrl

# dependencies

```bash
pip install numpy laspy open3d
pip install matplotlib
sudo apt-get install python3-tk
```

-[cloud_compare_to_visualise_data](https://flathub.org/en/apps/org.cloudcompare.CloudCompare) 
 

# run

```bash
cd ~/hacknation_cpk_ctrl/scripts
python3 interfejs.py

#then select file.las and start analyzing the data

#when finished, import generated .las data for example in CloudCompare 
```

# test generated .las

```bash
flatpak run org.cloudcompare.CloudCompare

#choose your new generate.las
```