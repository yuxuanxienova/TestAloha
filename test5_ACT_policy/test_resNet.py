import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import urllib.request
import io
import torchvision.transforms as transforms

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before sqrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

if __name__ == '__main__':
    # Load the pretrained model with FrozenBatchNorm2d
    model = torchvision.models.resnet18(pretrained=True, norm_layer=FrozenBatchNorm2d)
    model.eval()

    # Download an example image from the internet
    url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    response = urllib.request.urlopen(url)
    image_data = response.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        ),
    ])
    input_tensor = preprocess(image)# Dimension: (3, 224, 224)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # Move the input to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_batch = input_batch.to(device)

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_batch)

    # Print the output shape
    print('Output shape:', output.shape)
    # Expected output shape: [1, 1000] since ResNet outputs logits for 1000 classes

    # Optionally, get the predicted class
    _, predicted_class = torch.max(output, 1)
    print('Predicted class index:', predicted_class.item())