0: list level-2 CERES and MODIS file name

1: get both cloud albedo and COT for fields of successful retrievals (geo_utils.py and uniform_fov_tools.py are used)

2: get level-3 cloud albedo and COT

3: divide one file into each oceanic region a file

4: merge results from process 2 and 3, and filter invalid grid cell

build_sbdart_input/: transform surface albode and meteorological profiles from CERES and ERA5 to SBDART INPUT formation

fig1~fig4: plot codes

cal_yuan23_ac: reproduce cloud albedo in Yuan2023
