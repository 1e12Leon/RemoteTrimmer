import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision

class MySlimmingImportance(tp.importance.Importance):
    def __call__(self, group, **kwargs):
        # 1. First, define a list to store the importance of each layer in the group
        group_imp = [] # (num_bns, num_channels)
        # 2. Iterate through the layers in the group, compute importance for BN layers
        for dep, idxs in group: # idxs is a list of all prunable indices, useful for handling local coupling in DenseNet
            layer = dep.target.module # Get the nn.Module
            prune_fn = dep.handler    # Get the pruning function
            # 3. Compute importance for each BN layer
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and layer.affine:
                local_imp = torch.abs(layer.weight.data) # Compute the absolute value of the scale parameter
                group_imp.append(local_imp) # Store it in the list
        if len(group_imp) == 0: return None # Skip groups that don't contain BN layers
        # 4. Compute average importance across channels
        group_imp = torch.stack(group_imp, dim=0).mean(dim=0)
        return group_imp # (num_channels, )

# Create the model
name = "resnet18_delete_SE"
path = "models/" + name + ".pth"
model = torch.load(path)
# checkpoint = torch.load("model_save/resnet18_delete_SE_train_best.pth.tar")
# model.load_state_dict(checkpoint['state_dict'])
pruning_rate = 0.7

imp = MySlimmingImportance()

ignored_layers = []
for m in model.modules():
  if isinstance(m, torch.nn.Linear): # Ignore the classifier
    ignored_layers.append(m)

pruner = tp.pruner.GroupNormPruner(
    model = model,
    example_inputs = torch.randn(1, 3, 64, 64),
    importance = imp,     # Importance
    global_pruning=False, # Please refer to Page 9 of https://www.cs.princeton.edu/courses/archive/spring21/cos598D/lectures/pruning.pdf
    pruning_ratio = pruning_rate,    # Global sparsity for all layers
    #pruning_ratio_dict = {model.conv1: 0.2}, # Manually set the sparsity of model.conv1
    iterative_steps = 1,  # Number of steps to achieve the target channel sparsity
    ignored_layers = ignored_layers,        # Ignore some layers such as the final linear classifier
    # channel_groups = channel_groups,  # Round channels
    #unwrapped_parameters=[ (model.features[1][1].layer_scale, 0), (model.features[5][4].layer_scale, 0) ],
)

# Model size before pruning
base_macs, base_nparams = tp.utils.count_ops_and_params(model, torch.randn(1, 3, 64, 64))
pruner.step()

# Parameter & MACs Counter
pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, torch.randn(1, 3, 64, 64))
print("The pruned model:")
print(model)
print("Summary:")
print("Params: {:.2f} M => {:.2f} M".format(base_nparams/1e6, pruned_nparams/1e6))
print("MACs: {:.2f} G => {:.2f} G".format(base_macs/1e9, pruned_macs/1e9))

# Test Forward
output = model(torch.randn(1, 3, 64, 64))
print("Output.shape: ", output.shape)

torch.save(model, "models/resnet18_delete_SE(0.7).pth")
