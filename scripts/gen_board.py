import argparse
from cv2 import CV_8UC1
import numpy as np
import cv2 as cv
import os

def gen(args):

    MARKER_DICT = {
        'HD23': 6,
        'HD21': 12,
        'HD19': 38,
        'HD17': 157,
        'HD15': 766,
        'HD13': 2884,
        'HD11': 22335
    }

    texture_dir = ''
    total_markers = sum(args.n_markers)

    if args.family and MARKER_DICT[args.family] >= total_markers:
        texture_dir = os.path.join(args.texture_dir, args.family)
    else:
        for folder,n in MARKER_DICT.items():
            if total_markers <= n:
                texture_dir = os.path.join(args.texture_dir, folder)
                break

    if not texture_dir:
        print('No class contains {total_markers} unique markers.')

    pad = 0.1
    n_side_markers = int(args.n_markers[0]/4 + 1)
    inner_len = 800
    side_div_inner = n_side_markers + (n_side_markers - 1)*pad/(1 - 2*pad)
    side_len = int(side_div_inner*inner_len)
    i_marker = 0
    n = int(side_len + 2*pad*inner_len/(1 - 2*pad))
    out = 255*np.ones((n,n), dtype=np.uint8)

    p0 = 100

    for k,n_markers in enumerate(args.n_markers):

        n_side_markers = int(n_markers/4 + 1)
        side_div_inner = n_side_markers + (n_side_markers - 1)*pad/(1 - 2*pad)
        inner_len = int(side_len/side_div_inner)
        
        dp = int(inner_len*(1 - pad)/(1 - 2*pad))

        print(side_len)
        print(inner_len)
        print(dp)

        outer_inds = [0,n_side_markers-1]

        for i in range(n_side_markers):
            i_in_middle = i not in outer_inds
            for j in range(n_side_markers):
                if i_in_middle and j not in outer_inds:
                    continue
                
                img = cv.imread(os.path.join(texture_dir, f'{i_marker:05d}.png'), CV_8UC1)

                pi = p0 + i*dp
                pj = p0 + j*dp



                out[pi:pi+inner_len,pj:pj+inner_len] = cv.resize(
                    img[100:-100,100:-100], (inner_len,inner_len))

                i_marker += 1

        side_len -= 2*inner_len*(1 + pad/(1 - 2*pad))
        p0 += int(inner_len*(1 - pad)/(1 - 2*pad))

    if args.output:
        cv.imwrite(args.output, out)
    else:
        cv.imshow('Output', out)
        cv.waitKey()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate board of markers")

    parser.add_argument('n_markers', nargs='+', type=int,
        help='number of markers for each layer')
    parser.add_argument('-t', '--texture_dir', default='textures', 
        help='path to folder containing textures')
    parser.add_argument('-o', '--output', 
        help='path of output image')
    parser.add_argument('-f', '--family', 
        help='family of markers')

    gen(parser.parse_args())