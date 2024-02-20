import os, pickle, struct, traceback
import numpy as np
import cv2
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.io import read_image
from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim


# using DSSIM from https://github.com/VainF/pytorch-msssim/blob/master/tests/ae_example/train.py 
class DSSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(DSSIM_Loss, self).forward(img1, img2) )


def struct_unpack(data, counter, fmt):
    fmt_size = struct.calcsize(fmt)
    return (counter+fmt_size,) + struct.unpack (fmt, data[counter:counter+fmt_size])

class DFLJPG(object):
    """simplifed version of the DFLJPG class from DeepFaceLab, just to load and dump the mask"""
    def __init__(self, filename):
        self.filename = filename
        self.data = b""
        self.length = 0
        self.chunks = []
        self.dfl_dict = None
        self.shape = None
        self.img = None

    @staticmethod
    def load_raw(filename, loader_func=None):
        try:
            if loader_func is not None:
                data = loader_func(filename)
            else:
                with open(filename, "rb") as f:
                    data = f.read()
        except:
            raise FileNotFoundError(filename)

        try:
            inst = DFLJPG(filename)
            inst.data = data
            inst.length = len(data)
            inst_length = inst.length
            chunks = []
            data_counter = 0
            while data_counter < inst_length:
                chunk_m_l, chunk_m_h = struct.unpack("BB", data[data_counter:data_counter+2])
                data_counter += 2

                if chunk_m_l != 0xFF:
                    raise ValueError(f"No Valid JPG info in {filename}")

                chunk_name = None
                chunk_size = None
                chunk_data = None
                chunk_ex_data = None
                is_unk_chunk = False

                if chunk_m_h & 0xF0 == 0xD0:
                    n = chunk_m_h & 0x0F

                    if n >= 0 and n <= 7:
                        chunk_name = "RST%d" % (n)
                        chunk_size = 0
                    elif n == 0x8:
                        chunk_name = "SOI"
                        chunk_size = 0
                        if len(chunks) != 0:
                            raise Exception("")
                    elif n == 0x9:
                        chunk_name = "EOI"
                        chunk_size = 0
                    elif n == 0xA:
                        chunk_name = "SOS"
                    elif n == 0xB:
                        chunk_name = "DQT"
                    elif n == 0xD:
                        chunk_name = "DRI"
                        chunk_size = 2
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xC0:
                    n = chunk_m_h & 0x0F
                    if n == 0:
                        chunk_name = "SOF0"
                    elif n == 2:
                        chunk_name = "SOF2"
                    elif n == 4:
                        chunk_name = "DHT"
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xE0:
                    n = chunk_m_h & 0x0F
                    chunk_name = "APP%d" % (n)
                else:
                    is_unk_chunk = True

                #if is_unk_chunk:
                #    #raise ValueError(f"Unknown chunk {chunk_m_h} in {filename}")
                #    io.log_info(f"Unknown chunk {chunk_m_h} in {filename}")

                if chunk_size == None: #variable size
                    chunk_size, = struct.unpack(">H", data[data_counter:data_counter+2])
                    chunk_size -= 2
                    data_counter += 2

                if chunk_size > 0:
                    chunk_data = data[data_counter:data_counter+chunk_size]
                    data_counter += chunk_size

                if chunk_name == "SOS":
                    c = data_counter
                    while c < inst_length and (data[c] != 0xFF or data[c+1] != 0xD9):
                        c += 1

                    chunk_ex_data = data[data_counter:c]
                    data_counter = c

                chunks.append({'name' : chunk_name,
                                'm_h' : chunk_m_h,
                                'data' : chunk_data,
                                'ex_data' : chunk_ex_data,
                                })
            inst.chunks = chunks

            return inst
        except Exception as e:
            raise Exception (f"Corrupted JPG file {filename} {e}")

    @staticmethod
    def load(filename, loader_func=None):
        try:
            inst = DFLJPG.load_raw(filename, loader_func=loader_func)
            inst.dfl_dict = {}

            for chunk in inst.chunks:
                if chunk['name'] == 'APP0':
                    d, c = chunk['data'], 0
                    c, id, _ = struct_unpack(d, c, "=4sB")

                    if id == b"JFIF":
                        c, ver_major, ver_minor, units, Xdensity, Ydensity, Xthumbnail, Ythumbnail = struct_unpack(d, c, "=BBBHHBB")
                    else:
                        raise Exception("Unknown jpeg ID: %s" % (id) )
                elif chunk['name'] == 'SOF0' or chunk['name'] == 'SOF2':
                    d, c = chunk['data'], 0
                    c, precision, height, width = struct_unpack(d, c, ">BHH")
                    inst.shape = (height, width, 3)

                elif chunk['name'] == 'APP15':
                    if type(chunk['data']) == bytes:
                        inst.dfl_dict = pickle.loads(chunk['data'])

            return inst
        except Exception as e:
            print(f'Exception occured while DFLJPG.load : {traceback.format_exc()}')
            return None

    def has_data(self):
        return len(self.dfl_dict.keys()) != 0

    def save(self):
        try:
            with open(self.filename, "wb") as f:
                f.write ( self.dump() )
        except:
            raise Exception( f'cannot save {self.filename}' )

    def dump(self):
        data = b""

        dict_data = self.dfl_dict

        # Remove None keys
        for key in list(dict_data.keys()):
            if dict_data[key] is None:
                dict_data.pop(key)

        for chunk in self.chunks:
            if chunk['name'] == 'APP15':
                self.chunks.remove(chunk)
                break

        last_app_chunk = 0
        for i, chunk in enumerate (self.chunks):
            if chunk['m_h'] & 0xF0 == 0xE0:
                last_app_chunk = i

        dflchunk = {'name' : 'APP15',
                    'm_h' : 0xEF,
                    'data' : pickle.dumps(dict_data),
                    'ex_data' : None,
                    }
        self.chunks.insert(last_app_chunk+1, dflchunk)


        for chunk in self.chunks:
            data += struct.pack("BB", 0xFF, chunk['m_h'] )
            chunk_data = chunk['data']
            if chunk_data is not None:
                data += struct.pack(">H", len(chunk_data)+2 )
                data += chunk_data

            chunk_ex_data = chunk['ex_data']
            if chunk_ex_data is not None:
                data += chunk_ex_data

        return data

    def get_img(self):
        if self.img is None:
            self.img = read_image(self.filename)/255
        return self.img

    def get_shape(self):
        if self.shape is None:
            img = self.get_img()
            if img is not None:
                self.shape = img.shape
        return self.shape

    def get_dict(self):
        return self.dfl_dict

    def set_dict(self, dict_data=None):
        self.dfl_dict = dict_data

    def get_source_filename(self):                  return self.dfl_dict.get ('source_filename', None)
    def set_source_filename(self, source_filename): self.dfl_dict['source_filename'] = source_filename

    def get_source_rect(self):              return self.dfl_dict.get ('source_rect', None)
    def set_source_rect(self, source_rect): self.dfl_dict['source_rect'] = source_rect

    def get_source_landmarks(self):                     return np.array ( self.dfl_dict.get('source_landmarks', None) )
    def set_source_landmarks(self, source_landmarks):   self.dfl_dict['source_landmarks'] = source_landmarks

    def get_image_to_face_mat(self):
        mat = self.dfl_dict.get ('image_to_face_mat', None)
        if mat is not None:
            return np.array (mat)
        return None
    def set_image_to_face_mat(self, image_to_face_mat):   self.dfl_dict['image_to_face_mat'] = image_to_face_mat

    def has_xseg_mask(self):
        return self.dfl_dict.get('xseg_mask',None) is not None

    def get_xseg_mask_compressed(self):
        mask_buf = self.dfl_dict.get('xseg_mask',None)
        if mask_buf is None:
            return None

        return mask_buf
        
    def get_xseg_mask(self):
        mask_buf = self.dfl_dict.get('xseg_mask',None)
        if mask_buf is None:
            return None

        img = cv2.resize(cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED), self.get_shape()[:2])
        if len(img.shape) == 2:
            img = img[...,None]

        return torch.tensor(img, dtype=torch.float32 ).permute((2,0,1)) / 255.0

    def set_xseg_mask(self, mask_a):
        if mask_a is None:
            self.dfl_dict['xseg_mask'] = None
            return

        mask_a = imagelib.normalize_channels(mask_a, 1)
        img_data = np.clip( mask_a*255, 0, 255 ).astype(np.uint8)

        data_max_len = 50000

        ret, buf = cv2.imencode('.png', img_data)

        if not ret or len(buf) > data_max_len:
            for jpeg_quality in range(100,-1,-1):
                ret, buf = cv2.imencode( '.jpg', img_data, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality] )
                if ret and len(buf) <= data_max_len:
                    break

        if not ret:
            raise Exception("set_xseg_mask: unable to generate image data for set_xseg_mask")

        self.dfl_dict['xseg_mask'] = buf


class ImageSet(Dataset):
    """The simplest image loader possible"""
    def __init__(self, main_dir, masked = True, use_fp16=False):
        """Just take the folder path as input"""
        self.main_dir = main_dir
        self.masked = masked
        self.all_imgs = os.listdir(main_dir)
        self.use_fp16 = use_fp16

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        """returns Tuple (image,mask) 
        mask can be None"""
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        dflimg = DFLJPG.load(img_loc)

        image = dflimg.get_img()
        if self.use_fp16:
            image = image.half()

        if self.masked:
            if not dflimg.has_xseg_mask: raise AttributeError(f"No mask in {img_loc} !") 
            mask = dflimg.get_xseg_mask()
            if self.use_fp16: mask = mask.half() 
        else:
            mask = None

        return image, mask


class ImageSet_thomas(Dataset):
    """The simplest image loader possible"""
    def __init__(self, main_dir, masked = True, use_fp16=False):
        """Just take the folder path as input"""
        self.main_dir = main_dir
        self.masked = masked
        self.img_dst = os.listdir(main_dir+"/data_dst/aligned")
        self.img_src = os.listdir(main_dir+"/data_src/aligned")
        self.use_fp16 = use_fp16
        print(len(self.img_dst) + len(self.img_src))
    def __len__(self):
        return len(self.img_dst) + len(self.img_src) 

    def __getitem__(self, idx):
        if idx < len(self.img_dst):
            img_loc = os.path.join(self.main_dir+"/data_dst/aligned", self.img_dst[idx])
            label = torch.tensor(0)
        else:
            img_loc = os.path.join(self.main_dir+"/data_src/aligned", self.img_src[idx-len(self.img_dst)])
            label = torch.tensor(1)
        dflimg = DFLJPG.load(img_loc)
        image = dflimg.get_img()
        if self.use_fp16:
            image = image.half()

        if self.masked:
            if not dflimg.has_xseg_mask: raise AttributeError(f"No mask in {img_loc} !") 
            mask = dflimg.get_xseg_mask()
            if self.use_fp16: mask = mask.half() 
        else:
            mask = None
        return image.float() , label, mask