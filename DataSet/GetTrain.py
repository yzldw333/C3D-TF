import os,sys
import string


def load(imageinfo='val.txt',filename='',val_number=0,train_number=25000,min_number=25000):
    
    fread = open(imageinfo)
    indexset={}
    fw = open('gen_train'+filename+'.txt','w')
    fw_val = open('gen_val'+filename+'.txt','w')
    idx2images={}
    idx2num={}
    count = 0
    for line in fread:
        vec = line.strip().split()
        if len(vec) != 3:
            continue
        sku = vec[0]
        count += 1
        img = vec[0]+" "+str(count)
        idx = int(vec[2])
        result = "%s 0 %d\n"%(sku,idx)
        if idx not in idx2num:
            idx2num[idx] = 1
        else:
            idx2num[idx] += 1
        if idx2num[idx] <= val_number:
            fw_val.write(result) 
            continue
        if idx in idx2images:
            images = idx2images[idx]
            images[img]=result
            idx2images[idx] = images
        else:
            images={}
            images[img] = result
            idx2images[idx]=images
    for idx in idx2images:
        count = 0
        images = idx2images[idx]
        while(count < min_number - val_number):
            for sku in images:
                result = images[sku]
                count += 1
                if count > train_number - val_number:
                    break
                fw.write(result)
def main():

    #load('E:/dataset/VIVA_avi_group/VIVA_avi_part0/train.txt', '_new_', 0, 100, 100)
    #load('E:/dataset/VIVA_avi_group/VIVA_avi_part1/train.txt', '_new_', 0, 100, 100)
    #load('E:/dataset/VIVA_avi_group/VIVA_avi_part2/train.txt', '_new_', 0, 100, 100)
    #load('E:/dataset/VIVA_avi_group/VIVA_avi_part3/train.txt', '_new_', 0, 100, 100)
    #load('E:/dataset/VIVA_avi_group/VIVA_avi_part4/train.txt', '_new_', 0, 100, 100)
    #load('E:/dataset/VIVA_avi_group/VIVA_avi_part5/train.txt', '_new_', 0, 100, 100)
    #load('E:/dataset/VIVA_avi_group/VIVA_avi_part6/train.txt', '_new_', 0, 100, 100)
    load('E:/dataset/VIVA_avi_group/VIVA_avi_part7/train.txt', '_new_', 0, 100, 100)

    #genImage2id()

if __name__ == "__main__":
    main()
