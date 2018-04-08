def cp_files(fs, img_dir, mask_dir):
    #print(fs)
    for img, mask in fs:
        shutil.copy(img, img_dir)
        if mask_dir:
            shutil.copy(mask, mask_dir)

def generate_smpl_data():
    img_train_dir = join(args.TRAIN_DIR, 'X', 'img')
    mask_train_dir = join(args.TRAIN_DIR, 'y', 'mask')
    img_val_dir = join(args.VAL_DIR, 'X', 'img')
    mask_val_dir = join(args.VAL_DIR, 'y', 'mask')
    imgs = [join(img_train_dir, f) for f in listdir(img_train_dir)]
    masks = [join(mask_train_dir, f) for f in listdir(mask_train_dir)]
    imgs.sort()
    masks.sort()
    imgs_masks = list(zip(imgs, masks))
    random.shuffle(imgs_masks)

    train_smpl = imgs_masks[:100]
    val_smpl = imgs_masks[100:200]
    test_smpl = imgs_masks[200:300]
    cp_files(train_smpl, '/data/deepglobe/road/smpl/train/X/img/', '/data/deepglobe/road/smpl/tra
in/y/mask')
    cp_files(val_smpl, '/data/deepglobe/road/smpl/valid/X/img', '/data/deepglobe/road/smpl/valid/
y/mask')
    cp_files(test_smpl, '/data/deepglobe/road/smpl/test/', None)

def split_train_test():
    img_train_dir = join(args.TRAIN_DIR, 'X', 'img')
    mask_train_dir = join(args.TRAIN_DIR, 'y', 'mask')
    img_val_dir = join(args.VAL_DIR, 'X', 'img')
    mask_val_dir = join(args.VAL_DIR, 'y', 'mask')
    imgs = [join(img_train_dir, f) for f in listdir(img_train_dir)]
    masks = [join(mask_train_dir, f) for f in listdir(mask_train_dir)]
    imgs.sort()
    masks.sort()
    imgs_masks = list(zip(imgs, masks))
    random.shuffle(imgs_masks)
    
    valid_cutoff = int(len(imgs_masks) * .1)
    files_to_move = imgs_masks[:valid_cutoff]
    print(files_to_move)
    print(len(files_to_move))
    for img, mask in files_to_move:
        shutil.move(img, img_val_dir)
        shutil.move(mask, mask_val_dir)
