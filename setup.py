"""
RLHF with GRPO for ECG Medical Reasoning
=========================================

A project for training medical reasoning models using GRPO (Group Relative Policy Optimization)
on the ECG-Expert-QA dataset with the Llama-3.2-3B-Instruct base model.
"""

from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    # Core ML frameworks
    "torch==2.6.0",
    "torchvision==0.21.0",
    "transformers==4.56.0",
    "accelerate==0.34.2",
    "peft==0.13.0",

    # VERL/RL + telemetry (pin Ray; add Otel)
    "trl>=0.7.0",
    "ray[default]==2.9.0",
    "opentelemetry-api>=1.26,<2",
    "opentelemetry-sdk>=1.26,<2",
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "torchdata",
    "pyzmq>=25.0.0",          # Required for VERL vLLM rollout
    "vllm==0.8.4",            # Required for VERL rollout

    # Data processing
    "datasets>=2.14.0",
    "pandas==2.2.3",
    "numpy==1.26.4",
    "h5py==3.12.1",
    "joblib==1.4.2",
    "PyYAML==6.0.2",
    

    # Scientific computing
    "scipy==1.14.1",
    "scikit-learn==1.5.2",
    "scikit-image==0.21.0",

    # ECG-specific
    "ecg_plot @ git+https://github.com/willxxy/ecg-plot",
    "wfdb==4.1.2",

    # Deep learning utilities
    "einops==0.8.0",
    "PyWavelets==1.7.0",

    # Visualization
    "matplotlib==3.9.2",
    "seaborn==0.13.2",
    "tensorboard",

    # NLP & evaluation
    "nltk==3.9.1",
    "rouge==1.0.1",
    "sentencepiece==0.2.0",
    "sacremoses==0.1.1",
    "evaluate==0.4.3",
    "bert_score",
    "spacy==3.8.4",

    # Image processing
    "opencv_python==4.6.0.66",
    "pillow==10.3.0",
    "imageio==2.27.0",
    "imgaug==0.4.0",
    "imutils==0.5.4",

    # API & Web
    "openai>=1.70,<2",
    "requests==2.32.4",
    "gradio==5.31.0",
    "streamlit>=1.28.0",
    "beautifulsoup4==4.12.2",
    "html5lib==1.1",
    "validators==0.18.2",
    "qrcode==7.4.2",

    # ML interpretability
    "captum==0.7.0",

    # Utilities
    "tqdm==4.66.5",
    "wandb==0.18.3",
    "regex==2024.9.11",
    "codetiming",
    "hydra-core",
    "pybind11",
    "pylatexenc",
    "ninja==1.11.1.3",
    "termcolor==3.0.1",
    "faiss-cpu==1.10.0",

    # Build tools
    "maturin==1.7.4",

    # Development
    "ruff",
    "pre-commit",
    "pytest",
    "pydantic>=2.6,<3",
    "pydantic-core>=2.14,<3",

]

EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "isort>=5.10.0",
    ],
    "web": [
        "streamlit>=1.28.0",
        "gradio>=4.0.0",
    ],
    "flash": [
        "flash-attn==2.7.4.post1",
    ],
    "judge": [
        "llm-blender @ git+https://github.com/yuchenlin/LLM-Blender.git",
        "trl[judges]",
    ],
    "deepspeed": [
        "deepspeed>=0.12.0",  # Requires CUDA >= 12.1
    ],
}

# Keep "all" safe & buildable (omit flash-attn/deepspeed by default)
EXTRAS_REQUIRE["all"] = sorted(set(EXTRAS_REQUIRE["dev"] + EXTRAS_REQUIRE["web"] + EXTRAS_REQUIRE["judge"]))

setup(
    name="ecg-rlhf-grpo",
    version="0.1.0",
    author="Xiaoyu Song",
    author_email="sxysxysxysxy282828@gmail.com",
    description="RLHF with GRPO for ECG Medical Reasoning",
    long_description=open("README.md").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RL",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/RL/issues",
        "Documentation": "https://github.com/yourusername/RL/blob/main/README.md",
        "Source Code": "https://github.com/yourusername/RL",
    },

    packages=find_packages(exclude=["tests", "scripts", "data", "models", "outputs", "verl"]),
    py_modules=["prepare_data", "sft_train", "chat", "streamlit_chat", "analyze_data_samples"],

    python_requires=">=3.10",

    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    entry_points={
        "console_scripts": [
            "ecg-prepare-data=prepare_data:main",
            "ecg-sft-train=sft_train:main",
            "ecg-chat=chat:main",
            "ecg-streamlit-chat=streamlit_chat:main",
            "ecg-analyze-data=analyze_data_samples:main",
            # Optional: environment self-check command (see tools/env_check.py below)
            "ecg-check-env=tools.env_check:main",
        ],
    },

    include_package_data=True,
    package_data={"": ["*.txt", "*.md", "*.json", "*.sh"]},

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
    ],

    keywords=[
        "machine-learning",
        "deep-learning",
        "reinforcement-learning",
        "RLHF",
        "GRPO",
        "ECG",
        "medical-ai",
        "llama",
        "transformers",
        "pytorch",
    ],

    license="MIT",
    zip_safe=False,
)
