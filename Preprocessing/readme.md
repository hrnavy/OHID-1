### Image Fusion

Filename：Fusion.py

Function：Hyperspectral image fusion script

Variable Settings：

| Variable | Function                                         |
| -------- | ------------------------------------------------ |
| path     | Input path, specified to a folder                |
| out_path | Output path, specified to a folder               |
| bands    | Number of bands, currently 3 or 32 in common use |

Input Description:

```python
'''        
Input Description:
        Batch synthesis mode and single synthesis mode exist for this code:
        1. If you want to do batch image fusion for all sub-folders under a folder, you only need to fill in the name of the folder
            e.g. D:\\test folder has t1 folder and t2 folder under it, batch compositing just need to fill in r'D:\test\' in path.
        2. If you want to do image fusion for a single folder, just fill in the name of the folder.
            e.g. To synthesise bands for the folder D:\\test\\t1, just fill in r'D:\test\t1' in path.
         Regarding the input bands:
         1. To synthesise RGB image, just input list[R,G,B] in bands, RGB are the corresponding bands.
         	e.g. OHS data RGB bands are 15, 5, 1, to synthesise RGB image, you need to input [15,5,1] in bands.
         2. To composite multi-band images, enter list[b1,b2...,bn] in bands. ,bn].
         	e.g. OHS data has 32 bands, if you want to synthesise a 32-band tif image, you need to enter [1,2,.... ,32]
            
Special Notes:
   The code logic automatically determines whether there are sub-files under the folder, if there are sub-files, then single compositing mode is used by default;
   If there are only sub-folders without any files, batch compositing mode will be used.
'''
```

### image slice

Filename：slice.py

Function：Cutting large images into smaller ones

Variable Settings：

| Variable       | Function                           |
| -------------- | ---------------------------------- |
| img_path       | Input path, specified to a folder  |
| out_path       | Output path, specified to a folder |
| size_h，size_w | Slice Image Size                   |
| overlap        | overlap step                       |

Special Notes:
	When the size of the part of the image to be sliced is smaller than the slice size, the top and left sides of the sliced image are filled.

