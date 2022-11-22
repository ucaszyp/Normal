import numpy as np
import os
from PIL import Image
import threading
import random

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    try:
        score = correct.float().sum(0).mul(100.0 / correct.size(0))
        return score.item()
    except:
        return 0

def mIoU(output, target):
    """Computes the iou for the specified values of k"""
    num_classes = output.shape[1]
    hist = np.zeros((num_classes, num_classes))
    _, pred = output.max(1)
    pred = pred.cpu().data.numpy()
    target = target.cpu().data.numpy()
    hist += fast_hist(pred.flatten(), target.flatten(), num_classes)
    ious = per_class_iu(hist) * 100
    return round(np.nanmean(ious[1]), 2)

def mIoUAll(output, target):
    """Computes the iou for the specified values of k"""
    num_classes = output.shape[1]
    hist = np.zeros((num_classes, num_classes))
    _, pred = output.max(1)
    pred = pred.cpu().data.numpy()
    target = target.cpu().data.numpy()
    hist += fast_hist(pred.flatten(), target.flatten(), num_classes)
    ious = per_class_iu(hist) * 100
    return round(np.nanmean(ious), 2)


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #adjust the learning rate of sigma
    optimizer.param_groups[-1]['lr'] = lr * 0.01
    
    return lr

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = os.path.split(fn)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)

def save_colorful_images(predictions, filenames, output_dir, palettes):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    for ind in range(len(filenames)):
        im = Image.fromarray(palettes[predictions[ind].squeeze()])
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = os.path.split(fn)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)

def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return out


def random_crop(img, norm, norm_mask, height, width):
    """randomly crop the input image & surface normal
    """
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y + height, x:x + width, :]
    norm = norm[y:y + height, x:x + width, :]
    norm_mask = norm_mask[y:y + height, x:x + width, :]
    return img, norm, norm_mask


def color_augmentation(image, indoors=True):
    """color augmentation
    """
    # gamma augmentation
    gamma = random.uniform(0.9, 1.1)
    image_aug = image ** gamma

    # brightness augmentation
    if indoors:
        brightness = random.uniform(0.75, 1.25)
    else:
        brightness = random.uniform(0.9, 1.1)
    image_aug = image_aug * brightness

    # color augmentation
    colors = np.random.uniform(0.9, 1.1, size=3)
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    image_aug = np.clip(image_aug, 0, 1)
    return image_aug

def compute_normal_errors(total_normal_errors):
    metrics = {
        'mean': np.average(total_normal_errors),
        'median': np.median(total_normal_errors),
        'rmse': np.sqrt(np.sum(total_normal_errors * total_normal_errors) / total_normal_errors.shape),
        'a1': 100.0 * (np.sum(total_normal_errors < 5) / total_normal_errors.shape[0]),
        'a2': 100.0 * (np.sum(total_normal_errors < 7.5) / total_normal_errors.shape[0]),
        'a3': 100.0 * (np.sum(total_normal_errors < 11.25) / total_normal_errors.shape[0]),
        'a4': 100.0 * (np.sum(total_normal_errors < 22.5) / total_normal_errors.shape[0]),
        'a5': 100.0 * (np.sum(total_normal_errors < 30) / total_normal_errors.shape[0])
    }
    return metrics


# log normal errors
def log_normal_errors(metrics, where_to_write, first_line):
    print(first_line)
    print("mean median rmse 5 7.5 11.25 22.5 30")
    print("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (
        metrics['mean'], metrics['median'], metrics['rmse'],
        metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))

    with open(where_to_write, 'a') as f:
        f.write('%s\n' % first_line)
        f.write("mean median rmse 5 7.5 11.25 22.5 30\n")
        f.write("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n\n" % (
            metrics['mean'], metrics['median'], metrics['rmse'],
            metrics['a1'], metrics['a2'], metrics['a3'], metrics['a4'], metrics['a5']))