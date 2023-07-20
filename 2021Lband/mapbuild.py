''' Code to combine level5 individual dish and scan maps into a final level6 product
 - different combinantions of dishes and scans can be made for sub-set cross-correlation
 purposes.
 - maps can be saved by specifying an output directory
'''
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(1, '/idia/projects/hi_im/meerpower/meerpower')
import Init
import plot

####### AMEND BELOW TO LEVEL5 DATA VERSION ######
level5path = '/scratch3/users/jywang/MeerKLASS2021/level5/0.3/sigma4_count40/re_cali1_round5/'

halfway_dish=32
halfway_scan=15

dish_indx = np.arange(64)[:halfway_dish]
print(dish_indx)

scan_indx = np.arange(41)
print(scan_indx)

dish_indx = np.arange(4)
scan_indx = np.arange(4)

MKmap,counts = Init.subsetmap(level5path,dish_indx,scan_indx,verbose=True)

plt.imshow(np.mean(MKmap,2))
plt.colorbar()
plt.show()

exit()

###################################################################################################
# All available 2021 scans (observation block) and dishes (files don't exist for
#   all cases due to some failures):
scan = ['1630519596','1631379874','1631387336','1631552188','1631559762','1631659886',
            '1631667564','1631724508','1631732038','1631810671','1631818149','1631982988',
            '1631990463','1632069690','1632077222','1632184922','1632505883','1632760885',
            '1633365980','1633970780','1634252028','1634402485','1634748682','1634835083',
            '1637346562','1637354605','1637691677','1637699408','1638130295','1638294319',
            '1638301944','1638386189','1638639082','1638647186','1638898468','1639157507',
            '1639331184','1639935088','1640540184','1640712986','1640799689']
dish = []
for i in range(64):
    dish.append("%02d" %i)
###################################################################################################


# Print summary of available maps for analysing how splits should be made:
numberofscans,numberofdishes = len(scan),len(dish)
count = np.zeros((numberofscans,numberofdishes))
for n in range(numberofscans):
    for m in range(numberofdishes):
        filename = level5path + scan[n]+'_m0'+dish[m]+'_Sum_Tsky_xy_p0.3d.fits'
        if os.path.isfile(filename) is True: count[n,m] = 1

### Hard-coded halfway indices where to divide dishes and scans
halfway_dish = 32
halfway_scan = 15

# Splitting dishes in half, each sub-group (0<=m<halfway_dish) and (halfway_dish<=m<64):
print('--- Dish splitting:')
print(np.shape(count[:,:halfway_dish]))
print(np.shape(count[:,halfway_dish:]))
print(str(np.sum(count[:,:halfway_dish])) + ' maps in first group')
print(str(np.sum(count[:,halfway_dish:])) + ' maps in second group')

# Splitting scans in half, each sub-group (0<=n<halfway_scan) and (halfway_scan<=n<41):
print('--- Time-block splitting:')
print(np.shape(count[:halfway_scan,:]))
print(np.shape(count[halfway_scan:,:]))
print(str(np.sum(count[:halfway_scan,:])) + ' maps in first group')
print(str(np.sum(count[halfway_scan:,:])) + ' maps in second group')

dish = dish[:2]
scan = scan[:5]

numberofscans,numberofdishes = len(scan),len(dish)

MKmap_sum,counts_sum = None,None
for n in range(numberofscans):
    for m in range(numberofdishes):
        map_file = level5path + scan[n]+'_m0'+dish[m]+'_Sum_Tsky_xy_p0.3d.fits'
        counts_file = level5path + scan[n]+'_m0'+dish[m]+'_Npix_xy_count_p0.3d.fits'
        if os.path.isfile(map_file) is False: continue
        print(n,m)
        MKmap,w,W,counts = Init.ReadIn(map_file,counts_file,getcoord=False)

        if MKmap_sum is None: MKmap_sum = MKmap # for first map in loop
        else: MKmap_sum += MKmap
        if counts_sum is None: counts_sum = counts # for first map in loop
        else: counts_sum += counts

MKmap_ave = MKmap_sum/counts_sum

plt.imshow(np.mean(MKmap_ave,2))
plt.colorbar()
plt.show()

exit()
output_path = '/idia/projects/hi_im/raw_vis/MeerKLASS2021/subsetmaps/'
np.save(output_path+'dish0_%s_scan0-41'%halfway_dish,[dT_MK_dishesA,w_HI_dishesA,W_HI_dishesA,counts_dishesA])
#dT_MK_subA,w_HI_subA,W_HI_subA,counts_subA = np.load(filepath+'dishes0_%s.npy'%halfway_dish)
