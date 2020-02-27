## AR Tag Detection, Decoding and Imposition
The first part of the project involves detecting and decoding the AR tag code. The second part of the project involves imposing an 2-D image and 2-D cube onto the AR tag.

# AR tag:
![Reference AR Tag to be detected and tracked](data/reference_images/ref_marker.png)

# AR tag grid:
![Reference AR Tag in grid](data/reference_images/ref_marker_grid.png)

# Detection: 
Involves finding the outer corners of the AR tag from a video.
# Decoding:
Involves finding the AR code based on inner 2x2 grid of the AR tag.

# Imposing:
Involves superimposing Lena image and a virual 3-D cube onto the AR tag.

![Lena image](data/reference_images/Lena.png)

# Steps for finding the AR tag id:
```
cd code
python3 detection.py
```
# Sample output:
Output shown in the video frame:


![Output from Tag1 video](report/images/tag_id_outputvideo0.JPG)

Output shown in the AR tag with respect to orientation:

![Output from Tag1 video](report/images/warping_opencv.JPG)

# Steps for superimposing Lena image onto the AR tag
```
cd code
python3 imposing.py
```
# Sample output:

![Output from Tag1 video](report/images/Tag0_videooutput.JPG)

<a href="https://imgflip.com/gif/3qf5ez"><img src="https://i.imgflip.com/3qf5ez.gif" title="made at imgflip.com"/></a>

<a href="https://imgflip.com/gif/3qf5wq"><img src="https://i.imgflip.com/3qf5wq.gif" title="made at imgflip.com"/></a>

<a href="https://imgflip.com/gif/3qf602"><img src="https://i.imgflip.com/3qf602.gif" title="made at imgflip.com"/></a>

# Steps for superimposing virtual 3-D onto the AR tag
```
cd code
python3 virtual_cube_projection.py
```
# Sample output:
![Output from Tag2 video](report/images/Tag2_cube.JPG)
