# !/usr/bin/env python3
import numpy as np
import cv2
import random
from pathlib import Path
import argparse

def create_synthetic_image(w, h, t='random'):
    if t=='random':
        return np.random.randint(0,256,(h,w),dtype=np.uint8)
    if t=='gradient':
        g = np.linspace(0,255,w,dtype=np.uint8)
        return np.tile(g,(h,1))
    if t=='checkerboard':
        s=min(w,h)//8; img=np.zeros((h,w),dtype=np.uint8)
        for i in range(0,h,s):
            for j in range(0,w,s):
                if (i//s+j//s)%2==0:
                    img[i:i+s,j:j+s]=255
        return img
    if t=='circles':
        img=np.zeros((h,w),dtype=np.uint8)
        cx,cy=w//2,h//2; mr=min(w,h)//2
        y,x=np.ogrid[:h,:w]; d=np.sqrt((x-cx)**2+(y-cy)**2)
        for i in range(0,mr,mr//5):
            m=(d>=i)&(d<i+mr//10); img[m]=i*255//mr
        return img
    if t=='noise_bands':
        img=np.zeros((h,w),dtype=np.uint8); bh=h//4
        for i in range(4):
            nl=(i+1)*60
            img[i*bh:(i+1)*bh,:]=np.random.randint(0,nl,(bh,w),dtype=np.uint8)
        return img
    if t=='gaussian_blobs':
        img=np.zeros((h,w),dtype=np.uint8)
        for _ in range(random.randint(3,8)):
            cx,cy=random.randint(w//4,3*w//4),random.randint(h//4,3*h//4)
            sx,sy=random.randint(w//10,w//4),random.randint(h//10,h//4)
            I=random.randint(100,255)
            y,x=np.ogrid[:h,:w]
            b=I*np.exp(-((x-cx)**2/(2*sx**2)+(y-cy)**2/(2*sy**2)))
            img=np.maximum(img,b.astype(np.uint8))
        return img
    return np.random.randint(0,256,(h,w),dtype=np.uint8)

def generate_test_dataset(out, ns=15, nl=10):
    p=Path(out); p.mkdir(exist_ok=True)
    patterns=['random','gradient','checkerboard','circles','noise_bands','gaussian_blobs']
    for i in range(ns):
        w,h=random.choice([64,128,192,256]),random.choice([64,128,192,256])
        t=random.choice(patterns)
        cv2.imwrite(str(p/f"small_{i:03d}_{t}_{w}x{h}.png"),create_synthetic_image(w,h,t))
    for i in range(nl):
        w,h=random.choice([512,768,1024]),random.choice([512,768,1024])
        t=random.choice(patterns)
        cv2.imwrite(str(p/f"large_{i:03d}_{t}_{w}x{h}.png"),create_synthetic_image(w,h,t))
    cv2.imwrite(str(p/"tiny_001_gradient_32x32.png"),create_synthetic_image(32,32,'gradient'))
    try:
        cv2.imwrite(str(p/"huge_001_circles_2048x1536.png"),create_synthetic_image(2048,1536,'circles'))
    except MemoryError:
        pass
    hc=np.zeros((256,256),dtype=np.uint8); hc[:128,:]=255
    cv2.imwrite(str(p/"contrast_001_binary_256x256.png"),hc)
    lc=np.full((256,256),128,dtype=np.uint8)+np.random.randint(-10,11,(256,256))
    cv2.imwrite(str(p/"contrast_002_lowcontrast_256x256.png"),np.clip(lc,0,255).astype(np.uint8))
    with open(p/"dataset_info.txt",'w') as f:
        f.write(f"Small:{ns}\nLarge:{nl}\nSpecial:4\nPatterns:{','.join(patterns)}\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-o","--output",default="data")
    parser.add_argument("-s","--small",type=int,default=15)
    parser.add_argument("-l","--large",type=int,default=10)
    args=parser.parse_args()
    generate_test_dataset(args.output,args.small,args.large)

if __name__=="__main__":
    main()
