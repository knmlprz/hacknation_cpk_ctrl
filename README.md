# hacknation_cpk_ctrl

paste `file.las` to `~/hacknation_cpk_ctrl/scripts`

# dependencies

```bash
pip install laspy[lazrs] open3d numpy
pip install matplotlib
```

-[cloud_compare](https://flathub.org/en/apps/org.cloudcompare.CloudCompare) 
 

# run

```bash
cd ~/hacknation_cpk_ctrl/scripts
python3 segmentacja_na_pliki.py
```

# test generated .las

```bash
flatpak run org.cloudcompare.CloudCompare

#choose your new generate.las
```