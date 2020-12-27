import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
import os
from pathlib import Path
import sys

## importing our U-Net pytorch model
sys.path.append(str(Path("/isi/w/lb27/repos/phase_segmentation_pytorch")))
import Unet_model_multiclass_mini_adapted_BN as Unet_model

## boundary loss due to non-padded convolutions
FEATURE_PADS = {
    "down1.convr1": 2, "down1.convr2": 4, "down2.convr1": 10, "down2.convr2": 12, "down3.convr2": 26, 
    "down3.convr2": 28, "down4.convr1": 58, "down4.convr2": 60, "center.0": 122, "default": 124 }
## resizing to adjust for non-padded convolutions
## @BO: you can set this boolean as False
PADDED_RESIZE = True

def padded_resize(cam, layer_name, target_shape):
    if layer_name in FEATURE_PADS.keys():
        pad = FEATURE_PADS[layer_name]
    else:
        pad = FEATURE_PADS["default"]
    cam = cv2.resize(cam, (target_shape[0]-pad, target_shape[1]-pad))
    cam = np.pad(cam, pad_width=int(pad/2)) ## assumes padding is equal before and after, and also for all dimensions
    return cam

## Module for global average pooling
class AveragingModule(torch.nn.Module):
    def __init__(self):
        super(AveragingModule, self).__init__()
    def forward(self, x):
        return torch.mean(x, (2, 3))

## Module converting U-Net segmentation model to classification
class PhaseClassifier(torch.nn.Module):
    def __init__(self, feature_module):
        super(PhaseClassifier, self).__init__()
        self._feature_module = feature_module
        self._averaging_module = AveragingModule()
    def forward(self, x):
        x = self._feature_module(x)
        x = self._averaging_module(x)
        return x

## Original approach for feature extraction (not used anymore)
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.insert(0, grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        trace_list = []
        for name, module in self.model._modules.items():
            if "down" in name.lower():
                x, x_trace = module(x)
                trace_list.append(x_trace)
            elif "up" in name.lower():
                x_trace = trace_list.pop()
                x = module(x, x_trace)
            else:
                x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs.append(x)
        return outputs, x

## New approach for feature extraction
class FeatureExtractor_v2():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.activations = []
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_backward_hook(self.save_gradient)
                module.register_forward_hook(self.save_output)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.insert(0, grad_input)

    def save_output(self, module, input, output):
        self.activations.append(output)

    def __call__(self, x):
        x = self.model(x)
        return self.activations, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor_v2(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
                ## save out the probability map
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)        
        return target_activations, x

## Preprocessing image
def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()
    ## skipped preprocessing based on ImageNet data
    if False:
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

## Overlaying CAM mask on top of input image and saving the figure
def show_cam_on_image(img, mask, target_layer="", img_name="."):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img = np.dstack((img, img, img))
    cam = heatmap + np.float32(img)
    cam = cam / (np.max(cam) + sys.float_info.epsilon)
    cv2.imwrite(os.path.join(img_name, "cam_%s.jpg"%target_layer), np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.target_layer_names = target_layer_names

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        gradients = self.extractor.get_gradients()
        assert(len(gradients) == len(features))
        cam_list = []
        #grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        #target = features[-1]
        for gradient, feature, layer_name in zip (gradients, features, self.target_layer_names):
            grads_val = gradient[-1].cpu().data.numpy()
            target = feature.cpu().data.numpy()[0, :]
            weights = np.mean(grads_val, axis=(2, 3))[0, :]
            cam = np.zeros(target.shape[1:], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * target[i, :, :]

            cam = np.maximum(cam, 0)
            if PADDED_RESIZE:
                cam = padded_resize(cam, layer_name, input.shape[2:])
            else:
                cam = cv2.resize(cam, input.shape[2:])
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam) + sys.float_info.epsilon)
            cam_list.append(cam)
        return cam_list


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    # model = models.resnet50(pretrained=True)
    
    ## to load from an existing checkpoint
    ## @Bo: you can load your model from checkpoint here
    checkpoint_dir = Path("/isi/w/lb27/results/phase_segmentation/ray_exp00.dataset_selection/DEFAULT_8457c_00000_0_k=0_2020-11-09_18-10-50/checkpoint_247")
    path_cp = os.path.join(checkpoint_dir, "checkpoint")
    if not torch.cuda.is_available():
        checkpoint = torch.load(path_cp, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(path_cp)
    model = Unet_model.UNet_BN(bn=True)
    model.load_state_dict(checkpoint["net_state_dict"])
    print("loaded checkpoint: %s"%path_cp)
    
    ## The PhaseClassifier converts the segmentation output into a classification output by global average pooling
    ## This is required for gradcam to work
    phase_classifier = PhaseClassifier(model)
    ## all the target layers we visualized are a torch nn module comprising of a "Conv2d-BN-RELU" block
    ## Right now we assume that the target_layer_names contain layers in the increasing order of depth
    target_layer_names = ["down1.convr1", "down1.convr2", "down2.convr1", "down2.convr2", "down3.convr1", "down3.convr2", "down4.convr1", "down4.convr2",
                          "center.0", "center.1", 
                          "up1.convr1", "up1.convr2", "up2.convr1", "up2.convr2", "up3.convr1", "up3.convr2", "up4.convr1", "up4.convr2", "output_seg_map"]
    grad_cam = GradCam(model=phase_classifier, feature_module=phase_classifier._feature_module,
                        target_layer_names=target_layer_names, use_cuda=args.use_cuda)
    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    img_name = os.path.basename(args.image_path)
    try:
        os.mkdir(img_name)
    except:
        pass
    #img = np.float32(cv2.resize(img, (380, 380))) / 255
    img = img.astype(np.float32)/255
    input = preprocess_image(np.expand_dims(img, axis=-1))

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = 0
    mask_list = grad_cam(input, target_index)
    gb_model = GuidedBackpropReLUModel(model=phase_classifier, use_cuda=args.use_cuda)
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    for target_layer, mask in zip(target_layer_names, mask_list):
        show_cam_on_image(np.expand_dims(img, axis=-1), mask, target_layer, img_name)
        cam_mask = cv2.merge([mask, mask, mask])
        cam_gb = deprocess_image(cam_mask*gb)
        cv2.imwrite(os.path.join(img_name, 'cam_gb_%s.jpg'%target_layer), cam_gb)

    gb = deprocess_image(gb)
    cv2.imwrite(os.path.join(img_name, 'gb.jpg'), gb)

    if args.use_cuda:
        input = input.cuda()
    prediction = phase_classifier._feature_module(input)
    prediction = torch.sigmoid(prediction)
    prediction_np = prediction.detach().cpu().numpy()[0, :, :, :]*255
    cv2.imwrite(os.path.join(img_name, "prediction_bg.jpg"), prediction_np[0])
    cv2.imwrite(os.path.join(img_name, "prediction_bainite.jpg"), prediction_np[1])
    #bainite_np_padded = np.pad(prediction_np[1], pad_width=int((input.shape[-1]-prediction.shape[-1]/2)))
    #overlay_image = cv2.addWeighted(deprocess_image(input.detach().cpu().numpy()[0,0,:,:]), 0.7, deprocess_image(bainite_np_padded), 0.3, 0)
    #cv2.imwrite(os.path.join(img_name, "prediction_overlay.jpg"), overlay_image)