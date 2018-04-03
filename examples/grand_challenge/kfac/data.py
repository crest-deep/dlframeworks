import os
import tarfile
from PIL import Image
import numpy as np
from protonn.data.imaging.misc import set_size
from shuffle import shuffle
from timeit import default_timer as timer
from preprocessed_dataset import PreprocessedDataset
import chainermn
import chainer
from mpi4py import MPI


def list_dir_absolute(path):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    return files


def crop_center(img, crop_size=224):
    _, h, w = img.shape
    top = max((h - crop_size) // 2, 0)
    left = max((w - crop_size) // 2, 0)
    bottom = top + crop_size
    right = left + crop_size
    img = img[:, top:bottom, left:right]
    return img


def restore_channels(image):
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.shape[2] == 1:
        image = np.dstack([image, image, image])
    if image.shape[2] > 3:
        image = image[:, :, :3]
    return image


def read_image(path, dtype=np.float32, size_target=224, crop=False):
    with Image.open(path) as f:
        f.load()
        
        size_max_limit = 244
        size_min = min(f.size)
        #print(f"{path}, initial size: {f.size}, size_min = {size_min}")        
        # size_max = max(f.size) 
        if size_min > size_max_limit:
            #if crop:
            #    size_target = 228
            #else:
            #    size_target = size_max_limit
            size = ((f.size[0] * size_max_limit) // size_min, (f.size[1] * size_max_limit) // size_min)
            # print(f"shrinking to size {size}")
            f = f.resize(size, Image.LANCZOS)
        if  size_min < 224:
            size_target = 224
            size = ((f.size[0] * size_target) // size_min, (f.size[1] * size_target) // size_min)
            # print(f"enlarging to size {size}")
            f = f.resize(size, Image.LANCZOS)
        # f.save("/home/users/alex/projects/large_scale_ml/1.jpg")
        image = np.asarray(f, dtype=dtype)

        image = restore_channels(image)

            #image = set_size(image, (224, 224, 3))
#        except:
 #           print("error loading ", path)
        #finally:
        # Only pillow >= 3.0 has 'close' method
        #print(type(f))
        #if hasattr(f, 'close'):
            #f.close()
    image = image.transpose(2, 0, 1)
    # print("shape:", image.shape)
    image = crop_center(image, 256)
    #if crop:
        #image = crop_center(image)
    VGG_MEAN = [103.939, 116.779, 123.68]
    image[0, :, :] -= VGG_MEAN[0]
    image[1, :, :] -= VGG_MEAN[1]
    image[2, :, :] -= VGG_MEAN[2]
    #image -= 120.0
    image /= 200.0
    # print(image.shape)
    return image


def split_location(l, root):
    path, label = l.split()
    if root != ".":
        path = os.path.join(root, path)
    label = int(label)
    return path, label


def read_locations(path, root, cnt_classes=-1):
    with open(path) as file_location:
        locations = [split_location(line, root) for line in file_location]
    if cnt_classes > 0:
        locations = [l for l in locations if l[1] < cnt_classes]
    return locations


def read_dataset(locations):
    images = []
    labels = []
    for loc in locations:
        images.append(read_image(loc[0]))
        labels.append(loc[1])
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def read_archive(path_archive, size, crop):
    # images = []
    # labels = []
    data = []
    with tarfile.open(path_archive) as tar:
        file_locations = tar.extractfile("labels.txt").readlines()
        locations = [split_location(line.decode(), ".") for line in file_locations]
        for name_file_inner, label in locations:
            file_inner = tar.extractfile(name_file_inner)
            image = read_image(file_inner, size_target=size, crop=crop)
            #images.append(image)
            #labels.append(label)
            data.append((image, label))
    return data


def read_from_list_of_tars(tars, size, crop):
    # images = []
    # labels = []
    data = []
    for f in tars:
        data += read_archive(f, size, crop)
        # np.random.random((3, 224, 224)).astype(np.float32)
        # images += batch_images
        # labels += batch_labels
    return data


#def read_from_list_of_tars_as_np(tars):
    #images, labels = read_from_list_of_tars(tars)
    #return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def read_from_archves(comm, path, cnt_classes, size, crop):
    lst_tarfiles = []
    lst_tarfiles_local = []
    max_loaders = 500
    # ranks_with_files = comm.mpi_comm.allgather(comm.rank if ((comm.intra_rank == 0) and os.path.isdir(path)) else -1)
    ranks_with_files = comm.mpi_comm.allgather(comm.rank if os.path.isdir(path) else -1)
    ranks_with_files = [i for i in ranks_with_files if i >= 0]
    ranks_with_files.sort()
    # ranks_with_files = ranks_with_files[:max_loaders]
    if comm.rank == ranks_with_files[0]:
        print("loading tarfiles from", path)
        lst_tarfiles = list_dir_absolute(path)
        lst_tarfiles = [i for i in lst_tarfiles if int(i.split("_")[-2]) < cnt_classes]
        print(f"loaded {len(lst_tarfiles)} filenames, using {len(ranks_with_files)} readers")
    comm.mpi_comm.Barrier()

    count_me_in = comm.rank in ranks_with_files
    shuffle(lst_tarfiles, lst_tarfiles_local, comm.mpi_comm, pad=False, count_me_in=count_me_in)
    #if comm.rank < 6:

    if comm.rank == 0:
        print(f"---------------shuffled tarfiles to read, rank {comm.rank} got", len(lst_tarfiles_local))
        print("loading tars from storage")
    time_start = timer()
    if len(lst_tarfiles_local) > 0:
        data = read_from_list_of_tars(lst_tarfiles_local, size, crop)
    else:
        data = []
    comm.mpi_comm.Barrier()
    time_end = timer()
    if comm.rank == 0:
        print(f"done loading tars in {time_end - time_start}, shuffling")
    time_start = timer()

    # print(f"{comm.rank}: {lst_tarfiles_local}, {len(data)}")
    data_local = []
    shuffle(data, data_local, comm.mpi_comm, pad=True)
    comm.mpi_comm.Barrier()
    time_end = timer()
    if comm.rank == 0:
        print("done shuffling in", time_end - time_start)

    # for i in range(len(data_local)):
        # if data_local[i][0].shape != (3, 224, 224):
            # raise RuntimeError("wrong image loaded :", data_local[i][0].shape)

    return data_local


def read_by_list(comm, files, root, cnt_classes, size, random_crop):
    if comm.rank == 0:
        print("reading file list")
        locations = read_locations(files, root, cnt_classes)
        print("cnt samples global:", len(locations))
    else:
        locations = None
    locations = chainermn.scatter_dataset(locations, comm, shuffle=True)
    #print("cnt samples local:", len(locations))
    mean = np.ones((3, 256, 256), dtype=np.float32) * 128.0
    images = []
    for loc in locations:
        #if not os.path.isfile(loc[0]):
            #print(f"{MPI.Get_processor_name()}, missing {loc[0]}")
        img = read_image(loc[0])
        images.append((img, loc[1]))
    # images = chainer.datasets.LabeledImageDataset(locations, "./")
    ds = PreprocessedDataset(base=images, mean=mean, crop_size=224, random_crop=random_crop)
    #print(f"{MPI.Get_processor_name()}, done")
    #exit(-1)
    return ds


def read_data(comm, args, metadata):
    # if os.path.isfile(args.train):
    if args.train.endswith(".txt"):
        train = read_by_list(comm, args.train, args.root_train, args.cnt_classes, size=230, random_crop=True)
        val = read_by_list(comm, args.val, args.root_val, args.cnt_classes, size=230, random_crop=True)
            #train = PreprocessedDataset(args.train, args.root, mean, model.insize)
            #val = PreprocessedDataset(args.val, args.root, mean, model.insize, random=False)
        #else:
        #    train = None
        #    val = None
        #train = chainermn.scatter_dataset(train, comm, shuffle=True)
        #val = chainermn.scatter_dataset(val, comm)
    else:
        train = read_from_archves(comm, args.train, args.cnt_classes, size=230, crop=True)
        val = read_from_archves(comm, args.val, args.cnt_classes, size=230, crop=False)
    return train, val


def main():
    from tqdm import tqdm
    # print("testing")
    path_labels = "/home/share/ILSVRC2012/labels/train_abs.txt"
    locations = read_locations(path_labels, "/")
    for loc in tqdm(locations):
        img = read_image(loc[0])
        if img.shape[1]<224 or img.shape[2]<224: 
            print(loc[0], img.shape)
            break
        #print(img.shape)
    # print(location[:4])
    #images, labels = read_dataset(location[:10])
    # print(images[0].shape)
    # print(images[0][0])
    #path = "/tmp/imagenet"
    #tars = [os.path.join(path, f) for f in os.listdir(path)]
    #images, labels = read_from_list_of_tars(tars)
    #mages = np.array(images, dtype=np.float32)
    #labels = np.array(labels, dtype=np.int32)
    #for i in imahes:
        #print(i.shape)


if __name__ == "__main__":
    main()
