from setuptools import setup, find_packages

setup(
    name='runtime4yolo',
    version='1.0.0',
    packages=find_packages(),  # Automatically discovers sub-packages
    install_requires=[         # Add dependencies here
        'numpy',
        'pandas',
        'onnxruntime',
        'onnx',
        'openvino',
        'ultralytics',
        'tqdm'
        
    ],
    entry_points={
        'console_scripts': [
            'my_command = my_package.main:main_function',
        ],
    },
)