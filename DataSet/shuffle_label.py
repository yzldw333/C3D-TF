import random
def read_file_and_shuffle(infile,outfile):
    f = open(infile)
    lst = list(f)
    random.shuffle(lst)
    fw = open(outfile,'w')
    for e in lst:
        fw.write(e)
    fw.close()
    f.close()
if __name__ == '__main__':
    read_file_and_shuffle(r'E:\dataset\VIVA_avi_group\VIVA_avi_part0\gen_train_new_.txt',r'E:\dataset\VIVA_avi_group\VIVA_avi_part0\gen_train_shuffle.txt')
    read_file_and_shuffle(r'E:\dataset\VIVA_avi_group\VIVA_avi_part1\gen_train_new_.txt',r'E:\dataset\VIVA_avi_group\VIVA_avi_part1\gen_train_shuffle.txt')
    read_file_and_shuffle(r'E:\dataset\VIVA_avi_group\VIVA_avi_part2\gen_train_new_.txt',r'E:\dataset\VIVA_avi_group\VIVA_avi_part2\gen_train_shuffle.txt')
    read_file_and_shuffle(r'E:\dataset\VIVA_avi_group\VIVA_avi_part3\gen_train_new_.txt',r'E:\dataset\VIVA_avi_group\VIVA_avi_part3\gen_train_shuffle.txt')
    read_file_and_shuffle(r'E:\dataset\VIVA_avi_group\VIVA_avi_part4\gen_train_new_.txt',r'E:\dataset\VIVA_avi_group\VIVA_avi_part4\gen_train_shuffle.txt')
    read_file_and_shuffle(r'E:\dataset\VIVA_avi_group\VIVA_avi_part5\gen_train_new_.txt',r'E:\dataset\VIVA_avi_group\VIVA_avi_part5\gen_train_shuffle.txt')
    read_file_and_shuffle(r'E:\dataset\VIVA_avi_group\VIVA_avi_part6\gen_train_new_.txt',r'E:\dataset\VIVA_avi_group\VIVA_avi_part6\gen_train_shuffle.txt')
    read_file_and_shuffle(r'E:\dataset\VIVA_avi_group\VIVA_avi_part7\gen_train_new_.txt',r'E:\dataset\VIVA_avi_group\VIVA_avi_part7\gen_train_shuffle.txt')
    