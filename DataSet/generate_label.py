import os
import sys
import random
import random
choose_gestures = {1:0,2:1,3:2,4:3,6:4,7:5,8:6,13:7,14:8,15:9,16:10,21:11,23:12,27:13,28:14,29:15,30:16,31:17,32:18}
def generate_train_Label(dataroot,outdir):
    total_lst=[]
    lst = os.listdir(dataroot)
    fw_train = open(os.path.join(outdir,'train.txt'),'w')
    for e in lst:
        if os.path.isdir(os.path.join(dataroot,e))==True:
            man = (int(e.strip().split('_')[0])-1)/2
            label = int(e.strip().split('_')[1])
            if label==80:
                label=8
            if label==70:
                label=7
            if label in choose_gestures:
                label = choose_gestures[label]
            else:
                continue
            total_lst.append('%s 0 %s\n'%(e,label))
    random.shuffle(total_lst)
    train_lst = total_lst
    for e in train_lst:
        fw_train.write(e)
    fw_train.close()

def generate_val_Label(dataroot,outdir):
    total_lst=[]
    lst = os.listdir(dataroot)
    #fw_train = open(os.path.join(outdir,'train.txt'),'w')
    fw_val= open(os.path.join(outdir,'val.txt'),'w')
    for e in lst:
        if os.path.isdir(os.path.join(dataroot,e))==True:
            man = (int(e.strip().split('_')[0])-1)/2
            label = int(e.strip().split('_')[1])
            if label==80:
                label=8
            if label==70:
                label=7
            if label in choose_gestures:
                label = choose_gestures[label]
            else:
                continue
            total_lst.append('%s 0 %s\n'%(e,label))
    random.shuffle(total_lst)
    #length = len(total_lst)
    #train_lst = total_lst
    val_lst = total_lst
    #for e in train_lst:
    #    fw_train.write(e)
    for e in val_lst:
        fw_val.write(e)
    fw_val.close()

def generate_label(index):
    generate_train_Label(r'E:\dataset\VIVA_avi_group\VIVA_avi_part%d\train'%index,r'E:\dataset\VIVA_avi_group\VIVA_avi_part%d'%index)
    generate_val_Label(r'E:\dataset\VIVA_avi_group\VIVA_avi_part%d\val'%index,r'E:\dataset\VIVA_avi_group\VIVA_avi_part%d'%index)

if __name__ == '__main__':
    generate_label(0)
    generate_label(1)
    generate_label(2)
    generate_label(3)
    generate_label(4)
    generate_label(5)
    generate_label(6)
    generate_label(7)
