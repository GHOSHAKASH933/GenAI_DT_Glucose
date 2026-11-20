# Data

The code uses the ShanghaiT1DM and ShanghaiT2DM datasets (Zhao et al., *Scientific Data* 2023,
https://doi.org/10.1038/s41597-023-01940-7). They are published on figshare under a CC BY 4.0 licence
(https://doi.org/10.6084/m9.figshare.c.6310860) and are not copied into this repository.

Download the two collections and arrange them like this:

```
data/Shanghai_T1DM/*.xlsx        (12 patients, one file per monitoring record)
data/Shanghai_T2DM/*.xlsx        (100 patients)
data/Shanghai_T1DM_Summary.xlsx  (optional)
data/Shanghai_T2DM_Summary.xlsx  (optional)
```

`realdata/shanghai.py` reads these files, resamples every record to a 15-minute grid and builds the glucose,
insulin and meal channels. Run `python -m pytest` afterwards to check the setup.
