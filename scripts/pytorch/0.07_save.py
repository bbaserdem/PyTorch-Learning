import torch
import torchvision.models as models

DATA_DIR = "data/01_imagenet"
WEIGHTS_FILE = f"{DATA_DIR}/model_weights.pth"
MODEL_FILE = f"{DATA_DIR}/model_full.pth"

# Weights are returned by the state_dict method, and can be saved to files
model = models.vgg16(weights="IMAGENET1K_V1")
torch.save(model.state_dict(), WEIGHTS_FILE)

# To load a model weight, the same model needs to be loaded and the parameters need loading
model = models.vgg16()
model.load_state_dict(torch.load(WEIGHTS_FILE, weights_only=True))
model.eval()

print(model)

# The whole model with the architecture can be saved/loaded too
torch.save(model, MODEL_FILE)
model = torch.load(MODEL_FILE, weights_only=False)
# This uses python's pickle module, so relies on the actual class definition to be available
