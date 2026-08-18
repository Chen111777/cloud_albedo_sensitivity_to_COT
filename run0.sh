#!/bin/bash
python 0_1_list_modis.py 2020 1 west &
python 0_1_list_modis.py 2020 2 east &
python 0_1_list_modis.py 2020 2 west &
python 0_1_list_modis.py 2020 3 east &
python 0_1_list_modis.py 2020 3 west &
python 0_1_list_modis.py 2020 4 east &
python 0_1_list_modis.py 2020 4 west &
python 0_1_list_modis.py 2020 5 east &
python 0_1_list_modis.py 2020 5 west &
python 0_1_list_modis.py 2020 6 east &
python 0_1_list_modis.py 2020 6 west &
python 0_1_list_modis.py 2020 7 east &
python 0_1_list_modis.py 2020 7 west &
python 0_1_list_modis.py 2020 8 east &
python 0_1_list_modis.py 2020 8 west &
python 0_1_list_modis.py 2020 9 east &
python 0_1_list_modis.py 2020 9 west &
python 0_1_list_modis.py 2020 10 east &
python 0_1_list_modis.py 2020 10 west &
python 0_1_list_modis.py 2020 11 east &
python 0_1_list_modis.py 2020 11 west &
python 0_1_list_modis.py 2020 12 east &
python 0_1_list_modis.py 2020 12 west &

wait

echo "所有脚本执行完毕"
