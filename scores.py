import os

src_dir = "/nfsshare/home/xuyicheng/L2-MMD-GAN-CIFAR10/diff_dist_mmd_gan/data/jessicali9530/celeba-dataset/versions/2/img_align_celeba/img_align_celeba"
base_dir = "/nfsshare/home/xuyicheng/L2-MMD-GAN-CIFAR10/diff_dist_mmd_gan/data"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

files = sorted([
    f for f in os.listdir(src_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

n = len(files)
split = int(n * 0.9)

for f in files[:split]:
    src = os.path.join(src_dir, f)
    dst = os.path.join(train_dir, f)
    if not os.path.exists(dst):
        os.symlink(src, dst)

for f in files[split:]:
    src = os.path.join(src_dir, f)
    dst = os.path.join(val_dir, f)
    if not os.path.exists(dst):
        os.symlink(src, dst)

print("done")