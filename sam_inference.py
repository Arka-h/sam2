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
import logging
from ipyplot import plot_images
import math
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import random

def safe_state(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Load the Objects365 dataset
class Object365_Dataset(Dataset):
    def __init__(self, annots_dir, img_dir, transform=None, target_transform=None):
        self.annots_dir = annots_dir
        self.img_dir = img_dir
        self.transform = transform # tfm for the imgs
        self.target_transform = target_transform # tfm for the labels
        # Get the list of files
        self.data_list = sorted([ l[:-4] for l in os.listdir(annots_dir) if (l.endswith(".txt") and 
                                                                 (os.stat(
                                                                     os.path.join(self.annots_dir, l)
                                                                     ).st_size != 0) 
                                                                 )]) # Select only annots, ~ 14MB
        img_list = [ i for i in os.listdir(img_dir) if i.endswith(".jpg") ] # Select only jpgs, ~ 14MB
        # Data Validation
        assert len(self.data_list) > 0, "Wrong annotations path"
        assert len(img_list) > 0, "Wrong images path"
        
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

def StateFulDataLoader(DataLoader):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    # TODO: Check the idx processed in the batch, and save list of idx processed.
    # TODO: Check the savefile for the idx processed, and skip those idx.
    # TODO: Write a try catch block & write to IO, the bitmap of processed images.
    pass

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
    # Image embeddings are computed here
    return unroll_img_batch, unroll_bbox_batch, unroll_filename

def crop_img(unroll_sam_out, unroll_bbox_batch, new_dim=512, padding=0.06):
    padding = math.floor(padding*new_dim) # integer padding
    unroll_crop_out=[]
    for i, img in enumerate(unroll_sam_out):
        img = Image.fromarray(img)
        canvas = np.zeros((new_dim,new_dim,3), dtype=np.uint8)
        logging.debug(f"bbox: {unroll_bbox_batch[i]}")
        crop_img = img.crop(np.array(unroll_bbox_batch[i]))
        w_crop,h_crop = crop_img.size
        logging.debug(f"Image size: {w_crop}x{h_crop}")
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
        # logging.info(f"{colored_masks[i].shape}, {colored_masks[i].mean(axis=0).shape}, {img.shape}")
        temp = colored_masks[i].mean(axis=0).astype('bool')  # Convert to int, for mask operation | CWH
        # logging.info(f"Mask shape: {temp.shape}")
        temp = np.repeat(temp[np.newaxis, :, :], 3, axis=0) # Increase dim of the mask to 3 | 1WH->CWH
        logging.debug(f"Mask shape unroll: {temp.shape}, {img.shape}, {unroll_filename}")
        temp = temp * (img.cpu().detach().numpy()) # Apply mask to image
        temp = np.transpose(temp, (1,2,0)) # convert to HWC
        unroll_sam_out.append(temp)
    return unroll_sam_out

def run_inference(rank, world_size, predictor, logging, is_distributed, dset, bs):
    # Distributed computing code
    dist.init_process_group(rank=rank, world_size=world_size)
    predictor.to(rank)
    
    # distributed sampler
    sampler = DistributedSampler(dset) if is_distributed else None
    logging.info(f"len of dataset shard {dset.__len__()}") # img --> HWC, annot --> (N, 5)
    test_loader = StateFulDataLoader(dset, batch_size=bs, shuffle=False, collate_fn=collate_fn, sampler=sampler) 

    # Write a loop that iterates over the test_loader
    for data in tqdm(test_loader, desc="Processing dataset"):
        logging.debug(f"Unrolling {len(data[0])} instances")
        unroll_img_batch, unroll_bbox_batch, unroll_filename = unroll_data(*data)
        # Set the model to inference mode
        with torch.inference_mode():
            logging.info(f"Set image batch of {len(unroll_img_batch)} images")
            predictor.set_image_batch(unroll_img_batch) # Enter HWC
            logging.debug(f"Predicting Masks")
            colored_masks, scores, _ = predictor.predict_batch(box_batch=unroll_bbox_batch) # 
            logging.debug(f"Segment according to masks")
            unroll_sam_out = segment_data(unroll_img_batch, [colored_masks, unroll_filename]) # List of np arrays
            logging.debug(f"Cropping & resizing the images (default: 512)")
            unroll_crop_out = crop_img(unroll_sam_out, unroll_bbox_batch) # List of PIL images
            logging.debug(f"Saving the images")
            # Save the images
            for i, img in enumerate(unroll_crop_out):
                with open(f"/mnt/data/Objects365/processed/train/{unroll_filename[i]}.jpg", "wb") as f:
                    img.save(f)

# %%
if __name__=="__main__":
    # Arguments
    safe_state() # Set all the seeds
    logging.basicConfig(format='%(levelname)s | %(asctime)s.%(msecs)03d | %(module)s:%(lineno)d > %(message)s',datefmt='%H.%M.%S', level=logging.INFO)
    annots_dir = "/mnt/data/Objects365/labels/train/"
    img_dir = "/mnt/data/Objects365/images/train/"
    new_dim = 512
    bs=1
    world_size=2
    is_distributed=True
    # %%
    logging.info("Loading the dataset")
    # Load the dataset
    dset = Object365_Dataset(annots_dir, img_dir, img_tfm, xywh_rel_to_xyxy_abs)
    # Write the code for resuming capability here.
    logging.info("Instantiating sam2-hiera-large model")
    # Instantiate the sam model
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    # Run Inference 
    mp.spawn(run_inference, 
             args=(world_size, predictor, logging, is_distributed, dset, bs), 
             nprocs=world_size, join=True)
                    

'''
import pynvml

def get_available_gpu(min_free_memory=1024):  # min_free_memory in MB
    """Returns the ID of the first GPU with sufficient free memory, or None if none are available."""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for device_id in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory_mb = mem_info.free / 1024**2
        if free_memory_mb > min_free_memory:
            return device_id
    return None

def process_batches(model, dataloader):
    """Processes batches of data using a model, switching GPUs dynamically."""
    for batch_idx, batch_data in enumerate(dataloader):
        # Check available GPU before processing
        available_gpu = get_available_gpu(min_free_memory=1024)  # Set a threshold for free memory
        if available_gpu is None:
            print("No GPU available with sufficient memory. Exiting.")
            break
        
        # Transfer model and data to the selected GPU
        device = torch.device(f"cuda:{available_gpu}")
        model.to(device)
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        
        try:
            # Forward pass
            outputs = model(inputs)
            # Add your loss calculation, backward pass, and optimization here
            print(f"Batch {batch_idx} processed on GPU {available_gpu}")
        except RuntimeError as e:
            print(f"RuntimeError on GPU {available_gpu}: {e}")
            break

# Example Usage
# Assuming `model` is your PyTorch model and `dataloader` is a DataLoader
# Make sure the DataLoader provides batches of (inputs, labels)
model = ...  # Define or load your model
dataloader = ...  # Define your DataLoader
process_batches(model, dataloader)
'''
