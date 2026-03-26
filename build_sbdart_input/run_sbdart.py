# -*- coding: utf-8 -*-

import os
import re
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

def modify_input(tval=None, sz=None, ALBCON=None):
    """ modify parameters in INPUT file"""
    with open('INPUT', 'r') as f:
        content = f.read()
    if tval is not None and sz is None and ALBCON is None:

        pattern = r'(tcloud=)\s*(\d+(\.\d+)?|\.\d+)'
        modified_content = re.sub(
            pattern,
            f'tcloud={tval:.1f}',
            content
        )
    elif sz is not None and tval is None and ALBCON is None:
        pattern = r'(sza=)\s*(\d+(\.\d+)?|\.\d+)'
        modified_content = re.sub(
            pattern,
            f'sza={sz:.1f}',
            content
        )
    elif ALBCON is not None and tval is None and sz is None:
        print(ALBCON)
        pattern = r'(ALBCON=)\s*(\d+(\.\d+)?|\.\d+)'
        modified_content = re.sub(
            pattern,
            f'ALBCON={ALBCON:.4f}',
            content
        )
    else:
        print('order one modification at one time')
    # write in INPUT file
    with open('INPUT', 'w') as f:
        f.write(modified_content)
    
    


if __name__ == "__main__":
    # dcp, no seasonal and regional difference
    oceans = ['TPO']
    season_dict = {'MAM': [3, 4, 5]}
    # cp
    # oceans = ['NPO', 'NAO','TPO',  'TAO', 'TIO', 
    #           'SPO', 'SAO', 'SIO']
    # season_dict = {
    #     'MAM': [3, 4, 5],
    #     'JJA': [6, 7, 8],
    #     'SON': [9, 10, 11],
    #     'DJF': [12, 1, 2]
    # }
    
    target_link = Path('atms.dat')
    df_a = pd.read_csv("E:/cloud/251115_sbdart_A_cot/atms_dat_gascp/sfc_albedo_results.csv", index_col=0)
    
    for ocean in oceans:
        for season in season_dict.keys():
            source_file = Path(f'atms_dat_gasdcp/atms_{ocean}_{season}.dat')
            # delet old link
            if target_link.is_symlink() or target_link.exists():
                target_link.unlink()
            os.link(source_file, target_link)
            print(f"atms_dat → {source_file.name}")
            
            # modify_input(ALBCON=df_a.loc[ocean, season])
          
            tcloud_values = np.exp(np.linspace(np.log(0.03), np.log(180), 50))
            sz_values = np.arange(0, 76.1, 4)
            
            # initialization
            albedo_grid = np.full((len(sz_values), len(tcloud_values)), np.nan)
            
            # iterate through tcloud and sza, and fill albedo_grid
            for sz_idx, sz in enumerate(sz_values):
                print(f"Processing sz = {sz:.1f}...")
                modify_input(sz=sz)
                
                for tval_idx, tval in enumerate(tcloud_values):
                    modify_input(tval=tval)
                    
                    # run sbdart
                    subprocess.run('./sbdart.exe', check=True, capture_output=True, text=True)
                    
                    # read albedo from output
                    with open('out.txt', 'r') as f:
                        line = f.readline().strip()
                    parts = list(filter(None, line.split()))
                    val1 = float(parts[3])
                    val2 = float(parts[4])
                    albedo = val2 / val1
                    albedo_grid[sz_idx, tval_idx] = albedo
            
            # save as CSV
            df = pd.DataFrame(
                data=albedo_grid,
                index=sz_values,       # row index: sza
                columns=tcloud_values  # col index: tval
            )
            df.to_csv(f'sensitivity_dcp_cer13/cot_sza_to_albedo_lookup_table_{ocean}_{season}.csv', float_format='%.6f')
        
