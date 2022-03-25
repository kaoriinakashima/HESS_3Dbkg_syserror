import numpy as np
import yaml

with open("../general_config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
repo_path = cfg['repo_path']

# for gamma sources in the FoV
f = open(f'{repo_path}/{cfg["excl_regions_sources"]}', "r")
source_inFoV = []
for line in f.readlines():
    a = line.split('circle(')
    a = a[-1].split(') # ')
    a = a[0]
    coord = np.asarray(a.split(','), dtype=float)
    source_inFoV.append(coord)
source_inFoV = np.asarray(source_inFoV, dtype=float)

# for stars in the FoV
f = open(f'{repo_path}/{cfg["excl_regions_stars"]}', "r")
lines = f.readlines()
lines = lines[14:]
star_inFoV = []
for line in lines:
    if line[:17] == 'SEGMENT EX RADEC ':
        a = line.split()
        star_inFoV.append([a[3], a[4], a[7]])
star_inFoV = np.asarray(star_inFoV, dtype=float)


def get_mask_fov(SourcePosRA, SourcePosDEC, width):
    """this function will provide a mask of the gammaray sources and stars in a fov of diameter 'width',
    centered in a RA DEC position 'SourcePosRA' and 'SourcePosDEC'.
    All units of the input parameters are deg.
    Output:
        0 with there is no source nor star in this FoV
        else it will provide a list of regions to be excluded. In this list are: RA pos deg, DEC pos deg, radius of exclusion
    """
    st_mask = abs(star_inFoV.T[0] - SourcePosRA) <= width / 2
    st_mask &= abs(star_inFoV.T[1] - SourcePosDEC) <= width / 2
    so_mask = abs(source_inFoV.T[0] - SourcePosRA) <= width / 2
    so_mask &= abs(source_inFoV.T[1] - SourcePosDEC) <= width / 2
    if np.sum(st_mask) + np.sum(so_mask) == 0:
        return 0
    else:
        total_mask = []
        for i in star_inFoV[st_mask]:
            total_mask.append(i)
        for i in source_inFoV[so_mask]:
            total_mask.append(i)
        return total_mask
