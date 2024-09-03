import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision

class MySlimmingImportance(tp.importance.Importance):
    def __call__(self, group, **kwargs):
        # 1. 首先定义一个列表用于存储分组内每一层的重要性
        group_imp = [] # (num_bns, num_channels)
        # 2. 迭代分组内的各个层，对BN层计算重要性
        for dep, idxs in group: # idxs是一个包含所有可剪枝索引的列表，用于处理DenseNet中的局部耦合的情况
            layer = dep.target.module # 获取 nn.Module
            prune_fn = dep.handler    # 获取 剪枝函数
            # 3. 对每个BN层计算重要性
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and layer.affine:
                local_imp = torch.abs(layer.weight.data) # 计算scale参数的绝对值大小
                group_imp.append(local_imp) # 将其保存在列表中
        if len(group_imp)==0: return None # 跳过不包含BN层的分组
        # 4. 按通道计算平均重要性
        group_imp = torch.stack(group_imp, dim=0).mean(dim=0)
        return group_imp # (num_channels, )

# 创建网络模型
name = "resnet18_delete_SE"
path = "models/"+name+".pth"
model = torch.load(path)
# checkpoint = torch.load("model_save/resnet18_delete_SE_train_best.pth.tar")
# model.load_state_dict(checkpoint['state_dict'])
pruning_rate = 0.7

imp = MySlimmingImportance()

ignored_layers = []
for m in model.modules():
  if isinstance(m, torch.nn.Linear): # ignore the classifier
    ignored_layers.append(m)

pruner = tp.pruner.GroupNormPruner(
    model = model,
    example_inputs = torch.randn(1,3,64,64),
    importance = imp,     # Importance
    global_pruning=False, # Please refer to Page 9 of https://www.cs.princeton.edu/courses/archive/spring21/cos598D/lectures/pruning.pdf
    pruning_ratio = pruning_rate,    # global sparsity for all layers
    #pruning_ratio_dict = {model.conv1: 0.2}, # manually set the sparsity of model.conv1
    iterative_steps = 1,  # number of steps to achieve the target ch_sparsity.
    ignored_layers = ignored_layers,        # ignore some layers such as the finall linear classifier
    # channel_groups = channel_groups,  # round channels
    #unwrapped_parameters=[ (model.features[1][1].layer_scale, 0), (model.features[5][4].layer_scale, 0) ],
)

# Model size before pruning
base_macs, base_nparams = tp.utils.count_ops_and_params(model, torch.randn(1,3,64,64))
pruner.step()


# Parameter & MACs Counter
pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, torch.randn(1,3,64,64))
print("The pruned model:")
print(model)
print("Summary:")
print("Params: {:.2f} M => {:.2f} M".format(base_nparams/1e6, pruned_nparams/1e6))
print("MACs: {:.2f} G => {:.2f} G".format(base_macs/1e9, pruned_macs/1e9))

# Test Forward
output = model(torch.randn(1,3,64,64))
print("Output.shape: ", output.shape)

torch.save(model, "models/resnet18_delete_SE(0.7).pth")