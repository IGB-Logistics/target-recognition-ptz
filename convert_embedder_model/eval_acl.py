import acl
import numpy as np
import cv2
# export mobileNetV2_bottle as onnx model
from deep_sort_realtime.embedder.embedder_pytorch import MobileNetv2_Embedder as Embedder

# load pytorch model
embedder = Embedder(
    half=True,
    max_batch_size=16,
    bgr=True,
)
om_path = "/mnt/data/zhangyoujin/mobilenetv2/model/mobilenetv2.om"  # om模型文件的路径
img = cv2.imread("/mnt/data/zhangyoujin/mobilenetv2/dog.jpg")
img = embedder.preprocess(img)[0]
img = np.expand_dims(img, axis=0)
print("load data done.")

device_id = 0
NPY_FLOAT32 = 11
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_DEVICE, ACL_HOST = 0, 1
ACL_SUCCESS = 0

# ======运行管理资源申请======

# init acl resource
ret = acl.init()
if ret != ACL_SUCCESS:
    print('acl init failed, errorCode is', ret)

# 1.指定运算的Device。
ret = acl.rt.set_device(device_id)
print("set device ret:", ret, "device_id:", device_id)

# 2.显式创建一个Context，用于管理Stream对象。
context, ret = acl.rt.create_context(device_id)
print("create context ret:", ret)

# 3.显式创建一个Stream。
#用于维护一些异步操作的执行顺序，确保按照应用程序中的代码调用顺序执行任务。
stream, ret = acl.rt.create_stream()
print("create stream ret:", ret)

# load model from file
model_id, ret = acl.mdl.load_from_file(om_path)
if ret != ACL_SUCCESS:
    print('load model failed, errorCode is', ret)

# create description of model
model_desc = acl.mdl.create_desc()
ret = acl.mdl.get_desc(model_desc, model_id)
if ret != ACL_SUCCESS:
    print('get desc failed, errorCode is', ret)

# 2.准备模型推理的输入数据集。
# 创建aclmdlDataset类型的数据，描述模型推理的输入。
load_input_dataset = acl.mdl.create_dataset()
# 获取模型输入的数量。
input_size = acl.mdl.get_num_inputs(model_desc)
input_data = []
# 循环为每个输入申请内存，并将每个输入添加到aclmdlDataset类型的数据中。
for i in range(input_size):
    buffer_size = acl.mdl.get_input_size_by_index(model_desc, i)
    print("input buffer[", i, "] size:", buffer_size)
    # 申请输入内存。
    buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
    data = acl.create_data_buffer(buffer, buffer_size)
    _, ret = acl.mdl.add_dataset_buffer(load_input_dataset, data)
    print("add input dataset ret:", ret)
    input_data.append({"buffer": buffer, "size": buffer_size})

# 3.准备模型推理的输出数据集。
# 创建aclmdlDataset类型的数据，描述模型推理的输出。
load_output_dataset = acl.mdl.create_dataset()
# 获取模型输出的数量。
output_size = acl.mdl.get_num_outputs(model_desc)
output_data = []
# 循环为每个输出申请内存，并将每个输出添加到aclmdlDataset类型的数据中。
for i in range(output_size):
    buffer_size = acl.mdl.get_output_size_by_index(model_desc, i)
    print("output buffer[", i, "] size:", buffer_size)
    # 申请输出内存。
    buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
    data = acl.create_data_buffer(buffer, buffer_size)
    _, ret = acl.mdl.add_dataset_buffer(load_output_dataset, data)
    print("add output dataset ret:", ret)
    output_data.append({"buffer": buffer, "size": buffer_size})

# 2.申请内存后，可向内存中读入数据。
# 2.准备模型推理的输入数据，运行模式默认为运行模式为ACL_HOST，当前实例代码中模型只有一个输入。
bytes_data = img.tobytes()
np_ptr = acl.util.bytes_to_ptr(bytes_data)
# 将图片数据从Host传输到Device。
ret = acl.rt.memcpy(input_data[0]["buffer"], input_data[0]["size"], np_ptr,
                        input_data[0]["size"], ACL_MEMCPY_HOST_TO_DEVICE)
if ret != ACL_SUCCESS:
    print('memcpy failed, errorCode is', ret)
# 3.执行模型推理。
# self.model_id表示模型ID，在模型加载成功后，会返回标识模型的ID。
ret = acl.mdl.execute(model_id, load_input_dataset, load_output_dataset)
if ret != ACL_SUCCESS:
    print('memcpy failed, errorCode is', ret)


# 处理模型推理的输出数据，输出top5置信度的类别编号。
inference_result = []
for i, item in enumerate(output_data):
    buffer_host, ret = acl.rt.malloc_host(output_data[i]["size"])
    # 将推理输出数据从Device传输到Host。
    ret = acl.rt.memcpy(buffer_host, output_data[i]["size"], output_data[i]["buffer"],
                        output_data[i]["size"], ACL_MEMCPY_DEVICE_TO_HOST)
    bytes_out = acl.util.ptr_to_bytes(buffer_host, output_data[i]["size"])
    data = np.frombuffer(bytes_out, dtype=np.float32)
    inference_result.append(data)
    print("data:", data)
    

# 释放模型推理的输入、输出资源。
# 释放输入资源，包括数据结构和内存。
while input_data:
    item = input_data.pop()
    ret = acl.rt.free(item["buffer"])
input_number = acl.mdl.get_dataset_num_buffers(load_input_dataset)
for i in range(input_number):
    data_buf = acl.mdl.get_dataset_buffer(load_input_dataset, i)
    if data_buf:
        ret = acl.destroy_data_buffer(data_buf)
ret = acl.mdl.destroy_dataset(load_input_dataset)
print("destroy input dataset ret:", ret)

# 释放输出资源，包括数据结构和内存。
while output_data:
    item = output_data.pop()
    ret = acl.rt.free(item["buffer"])
output_number = acl.mdl.get_dataset_num_buffers(load_output_dataset)
for i in range(output_number):
    data_buf = acl.mdl.get_dataset_buffer(load_output_dataset, i)
    if data_buf:
        ret = acl.destroy_data_buffer(data_buf)
ret = acl.mdl.destroy_dataset(load_output_dataset)
print("destroy output dataset ret:", ret)

#......

# ======运行管理资源释放======
ret = acl.rt.destroy_stream(stream)
print("destroy stream ret:", ret)
ret = acl.rt.destroy_context(context)
print("destroy context ret:", ret)
ret = acl.rt.reset_device(device_id)
print("reset device ret:", ret)
# ======运行管理资源释放======

#......
