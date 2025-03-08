# %%
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging as log
from ipyplot import plot_images
import math
import traceback
import argparse

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# Load the Objects365 dataset
class Object365_Dataset(Dataset):
    def __init__(self, annots_dir, img_dir, target_dir, transform=None, target_transform=None, resume=True):
        self.annots_dir = annots_dir
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.transform = transform # tfm for the imgs
        self.target_transform = target_transform # tfm for the labels
        # Get the list of files
        self.data_list = [ l[:-4] for l in os.listdir(annots_dir) if (l.endswith(".txt") and 
                                                                 (os.stat(
                                                                     os.path.join(self.annots_dir, l)
                                                                     ).st_size != 0) 
                                                                 )] # Select only jpgs, ~ 14MB
        img_list = [ i for i in os.listdir(img_dir) if i.endswith(".jpg") ] # Select only jpgs, ~ 14MB
        log.info(f"Found {len(self.data_list)} annotations")
        # Data Validation
        assert len(self.data_list) > 0, "Wrong annotations path"
        assert len(img_list) > 0, "Wrong images path"
        
        if resume: # Remove Repetition;    data: objects365_v2_01692748_11_0.jpg -> 5 parts, need first 3
            log.info(f"Removing processed files from the list")
            processed_list = [ i[:-4].split("_") for i in os.listdir(target_dir) if i.endswith(".jpg") ] # (N, 5)
            processed_list = set([ "_".join(i[:3])for i in processed_list ]) # Remove the last two parts, remove duplicates
            self.data_list = list(filter(lambda x: x not in processed_list, self.data_list)) # Remove already processed files
            log.info(f"Found {len(self.data_list)} annotations")

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx): # samples from the dataset at single random index idx
        img_path = os.path.join(self.img_dir, self.data_list[idx]+".jpg")
        img = read_image(img_path) # default CHW array
        annot_path = os.path.join(self.annots_dir, self.data_list[idx]+".txt")
        annot = torch.Tensor(np.loadtxt(annot_path)).reshape(-1, 5) # (N, 5) tensor
        filename = self.data_list[idx]
        # Apply transformations
        if self.transform: img = self.transform(img)
        if self.target_transform: annot = self.target_transform(annot, img) # transform to the sam2 format
        return [img, annot, filename] # single img, single MD-annot, single filename
    
def img_tfm(image): # Stays CHW
    # image = torch.permute(image, (1,2,0)) 
    return image

def xywh_rel_to_xyxy_abs(annot, image):
    _,w,h = image.shape # CWH
    # Convert relative to absolute
    annot[:,1::2], annot[:,2::2] = annot[:,1::2]*w, annot[:,2::2]*h
    # Convert xywh to xyxy (SAM's format)
    annot[:,1] = torch.round(annot[:,1] - annot[:,3]/2)
    annot[:,2] = torch.round(annot[:,2] - annot[:,4]/2)
    annot[:,3] = torch.round(annot[:,1] + annot[:,3]/2)
    annot[:,4] = torch.round(annot[:,2] + annot[:,4]/2)
    
    return annot.type(torch.int32)
    
def collate_fn(batch): # Gets a batch of data from the loader
    batch_img = [item[0] for item in batch]
    batch_annot = [item[1] for item in batch]
    batch_filename = [item[2] for item in batch]
    return batch_img, batch_annot, batch_filename

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2)) 
    
def show_masks(image, masks, scores, box_coords=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def write_img_annot_stats(annots_dir, img_dir):
    with open("annots.txt", "w") as f:
        filename = [l[:-4] for l in os.listdir(annots_dir) if l.endswith(".txt")]
        filename.sort()
        f.write("\n".join(filename))
    with open("imgs.txt", "w") as f:
        filename = [l[:-4] for l in os.listdir(img_dir) if l.endswith(".jpg")]
        filename.sort()
        f.write("\n".join(filename))
    print("Done writing the stats")
    os.system("diff annots.txt imgs.txt") # Shows the difference between the two files

def unroll_data(batch_img, batch_annot, batch_filename):
    n = len(batch_annot) # number of items in the batch
    unroll_img_batch = []
    unroll_bbox_batch = []
    unroll_filename = []
    unroll_count = [annot.shape[0] for annot in batch_annot]
    # Unroll the images, labels to ingest into model
    for i in range(n): 
        unroll_img_batch.extend([batch_img[i]]*unroll_count[i])
        unroll_bbox_batch.extend([bbox[1:] for bbox in batch_annot[i]])
        unroll_filename.extend([f"{batch_filename[i]}_{bbox[0]}_{j}" for j, bbox in enumerate(batch_annot[i])])
    # Check for empty bbox, and remove them
    temp = torch.stack(unroll_bbox_batch)
    is_empty = (temp[:, 0]==temp[:, 2])|(temp[:,1]==temp[:,3])
    if torch.any(is_empty):
        idx = torch.where(is_empty)[0] # torch.where() Always returns a tuple of tensor with multiple indices
        log.info(f"Empty bbox found in {[unroll_filename[i] for i in idx]}")
        unroll_img_batch = [element for i, element in enumerate(unroll_img_batch) if i not in idx ]
        unroll_bbox_batch = [element for i, element in enumerate(unroll_bbox_batch) if i not in idx ]
        unroll_filename = [element for i, element in enumerate(unroll_filename) if i not in idx ]
        log.info(f"bbox(s) not processed")
    return unroll_img_batch, unroll_bbox_batch, unroll_filename

def crop_img(unroll_sam_out, unroll_bbox_batch, new_dim=512, padding=0.06):
    padding = math.floor(padding*new_dim) # integer padding
    unroll_crop_out=[]
    for i, img in enumerate(unroll_sam_out):
        img = Image.fromarray(img)
        canvas = np.zeros((new_dim,new_dim,3), dtype=np.uint8)
        log.debug(f"bbox: {unroll_bbox_batch[i]}")
        crop_img = img.crop(np.array(unroll_bbox_batch[i]))
        w_crop,h_crop = crop_img.size
        log.debug(f"Image size: {w_crop}x{h_crop}")
        pad_dim = new_dim-padding # Automatically pads the image
        if w_crop>h_crop:
            w_new, h_new = pad_dim, int(h_crop*(pad_dim/w_crop))
            new_img = crop_img.resize((w_new, h_new)) # Resize the image to match new_dim
            canvas[new_dim//2-h_new//2:
                new_dim//2+h_new//2+(h_new&1),
                padding//2:
                    -(padding//2+(padding&1)),
                    :] = new_img
        else: # h occupies the axis=0
            w_new, h_new = int(w_crop*(pad_dim/h_crop)), pad_dim
            new_img = crop_img.resize((w_new, h_new))
            canvas[padding//2:
                -(padding//2+(padding&1)),
                new_dim//2-w_new//2:
                    new_dim//2+w_new//2+(w_new&1),
                    :] = new_img
        unroll_crop_out.append(Image.fromarray(canvas))
    return unroll_crop_out

def segment_data(unroll_img_batch, colored_masks):
    colored_masks, unroll_filename = colored_masks
    unroll_sam_out = []
    for i, img in enumerate(unroll_img_batch):
        # log.info(f"{colored_masks[i].shape}, {colored_masks[i].mean(axis=0).shape}, {img.shape}")
        temp = colored_masks[i].mean(axis=0).astype('bool')  # Convert to int, for mask operation | CWH
        # log.info(f"Mask shape: {temp.shape}")
        temp = np.repeat(temp[np.newaxis, :, :], 3, axis=0) # Increase dim of the mask to 3 | 1WH->CWH
        log.debug(f"Mask shape unroll: {temp.shape}, {img.shape}, {unroll_filename}")
        temp = temp * (img.cpu().detach().numpy()) # Apply mask to image
        temp = np.transpose(temp, (1,2,0)) # convert to HWC
        unroll_sam_out.append(temp)
    return unroll_sam_out
# %%
if __name__=="__main__":
    """
    # Manual calculations of data speed:
    6410 batches of 8 images each with average image size of (664 GB)/1742391 ~ 400 KB
    6000 Ada: 6410*8*400/(1024**2) ~ 20 GB of input images processed in 40 hrs.
    20 * (48/40) ~ 24 GB of input images processed in 48 hrs. (Hence we shall store 25GB of images in each directory)

    Q. So how many files per user to dump during a training run?
    > 25*1024**2/400 = 65536 --> 65600
    """
    uname = [
        "arkahaldi", "akanksha1", "dharmasai", "rishig", "sekhar", "shyam.marjit"
    ] # 6 x 25 = 150GB processed at a time
    chunk_sz = 65600
    p = argparse.ArgumentParser(description="Inference pipeline[SAM2] for localization")
    p.add_argument("--log", help="log level", default="INFO")
    args = p.parse_args()
    log.basicConfig(format='%(levelname)s | %(asctime)s.%(msecs)03d | %(module)s:%(lineno)d > %(message)s',datefmt='%H.%M.%S', level=log.getLevelName(args.log))
    # %%
    log.info("Loading the dataset")
    # Load the dataset
    annots_dir = "/mnt/data/Objects365/labels/train/"
    img_dir = "/mnt/data/Objects365/images/train/"
    target_dir = "/mnt/data/Objects365/processed/train/"
    new_dim = 512
    dl_bs=8
    bs=16
    # write_img_annot_stats(annots_dir, img_dir)
    dset = Object365_Dataset(annots_dir, img_dir, target_dir, img_tfm, xywh_rel_to_xyxy_abs)
    log.info(f"len of dataset {dset.__len__()}") # img --> HWC, annot --> (N, 5)
    log.info(f"dl_bs: {dl_bs}, batch_size: {bs}")
    test_loader = DataLoader(dset, batch_size=dl_bs, shuffle=False, collate_fn=collate_fn) # TODO: Save the generator state in dataloader, Write your own sampler and pass to DataLoader.
    # Write the code for resuming capability here.
    log.info("Instantiating sam2-hiera-large model")
    # Instantiate the sam model
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    predictor.model.to("cuda:1")
    # Write a loop that iterates over the test_loader
    processed_file = '' # Use for cleanup
    try:
        for data in tqdm(test_loader, desc="Processing dataset (img, annot)"):
            unroll_img_batch, unroll_bbox_batch, unroll_filename = unroll_data(*data) # Unroll the data to feed into the model
            log.debug(f"Unrolled into {len(unroll_img_batch)} instances")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                # Shard output into smaller batches
                for s in tqdm(range(0, len(unroll_img_batch), bs), desc="Processing shard (unrolled)"):
                    unroll_img_shard, unroll_bbox_shard, unroll_file_shard = unroll_img_batch[s:s+bs], unroll_bbox_batch[s:s+bs], unroll_filename[s:s+bs]
                    log.debug(f"Setting Image batch")
                    predictor.set_image_batch(unroll_img_shard) # Set the image batch
                    log.debug(f"Predicting Masks")
                    colored_masks, scores, _ = predictor.predict_batch(box_batch=unroll_bbox_shard) # 
                    log.debug(f"Segment according to masks")
                    unroll_sam_out = segment_data(unroll_img_shard, [colored_masks, unroll_file_shard]) # List of np arrays
                    log.debug(f"Cropping & resizing the images (default: 512)")
                    unroll_crop_out = crop_img(unroll_sam_out, unroll_bbox_shard) # List of PIL images
                    log.debug(f"Saving the images")
                    # Save the images
                    for i, img in enumerate(unroll_crop_out):
                        with open(f"{target_dir}/{unroll_file_shard[i]}.jpg", "wb") as f:
                            img.save(f)
                            processed_file = unroll_file_shard[i]
                    # log.debug(f"Processed {unroll_file_shard}")
    except KeyboardInterrupt:
        log.info("Process interrupted, cleaning up files...")
        processed = processed_file.split("_")
        filename = "_".join(processed[:3])
        n=0
        with open(os.path.join(annots_dir, filename+".txt"), 'r') as f: n = len(f.readlines())
        if n > (int(processed[4])+1): # 5th element is the line number in the annot file
            os.system(f"rm {target_dir}/{filename}*.jpg") # Remove partially processed files; # to include in next dataset
        exit(0)
    except Exception as e:
        log.info(f"Error occurred on file: {processed_file}")
        log.error(f"Error: {e}")
        traceback.print_exc()
        exit(1)