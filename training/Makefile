prereq:
	pip install git+https://github.com/KhronosGroup/NNEF-Tools.git
	pip install future typing six numpy protobuf onnx torch
	pip install future typing six numpy protobuf
MNIST:
	python main_mnist.py --size 1 --permute 1
	cd training_data
	python ../convert.py --input-format onnx --output-format nnef --input-mode "FC_Small.onnx" --output-mode "FC_Small.nnef"
CIFAR:
	python main_conv.py --size 1 --permute 1 --dataset CIFAR10
	cd training_data
	python ../convert.py --input-format onnx --output-format nnef --input-mode "CNN_Large.onnx" --output-mode "CNN_Large.nnef"
