export ASCEND_RT_VISIBLE_DEVICES=5
python - <<'PY'
import torch
import torch_npu

print("device_count =", torch.npu.device_count())
torch.npu.set_device(0)
x = torch.tensor([1.0, 2.0]).npu()
print(x)
PY